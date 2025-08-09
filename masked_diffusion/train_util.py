import copy
import functools
import os
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import autocast
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from diffusers.models import AutoencoderKL
from adan import Adan
from torch.distributed.optim import ZeroRedundancyOptimizer

INITIAL_LOG_LOSS_SCALE = 20.0

class DINOv2WithHiddenStates(th.nn.Module):    
    """
    DINOv2WithHiddenStates.

    Args:
        x (torch.Tensor):
            Input image tensor.
            Shape: (batch_size, channels, height, width)
            Example: (B, 3, 224, 224)

    Returns:
        List[torch.Tensor]:
            List of hidden states from all transformer blocks in the original model.
            Each element has shape:
                (batch_size, num_patches + 1, embedding_dim)
            Example: [(B, 257, 1024), ..., (B, 257, 1024)]
            - The +1 in num_patches accounts for the CLS token.
    """
    def __init__(self, original_model):
        super(DINOv2WithHiddenStates, self).__init__()
        self.original_model = original_model
        self.hidden_states = []

    def hook_fn(self, module, input, output):
        self.hidden_states.append(output)

    def forward(self, x):
        self.hidden_states = []  # Clear previous states
        hooks = []

        # Register hooks on layers of interest
        for layer in self.original_model.blocks:  # Assuming 'blocks' contains transformer layers
            hook = layer.register_forward_hook(self.hook_fn)
            hooks.append(hook)

        # Forward pass
        _ = self.original_model(x)

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        return self.hidden_states  # Return the hidden states

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        scale_factor=0.18215, # scale_factor follows DiT and stable diffusion.
        opt_type='adan',
        use_zero=True, 
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.scale_factor = scale_factor

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        if opt_type=='adamw':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=Adam,
                    lr=self.lr,
                     weight_decay=self.weight_decay
                )
            else:
                self.opt = AdamW(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
                )
        elif opt_type=='adan':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=Adan,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    max_grad_norm=1, fused=True
                )
                
            else:
                self.opt = Adan(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay, max_grad_norm=1, fused=True)
        
        for group in self.opt.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = param.grad.to(dist_util.dev())
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.instantiate_first_stage()
        
        self.instantiate_image_encoder()

    def instantiate_first_stage(self):

        model = AutoencoderKL.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",subfolder="vae").to(dist_util.dev()) 
        self.first_stage_model = model.eval()
        self.first_stage_model.train = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_image_encoder(self):
        th.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.image_encoder = th.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(dist_util.dev())
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    @th.no_grad()
    def get_first_stage_encoding(self, x):
            encoder_posterior = self.first_stage_model.encode(x).latent_dist.sample()     
            z = encoder_posterior   
            return z.to(dist_util.dev()) * self.first_stage_model.config.scaling_factor

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        for name, param in self.model.named_parameters():
            param.data = param.data.to(dist_util.dev())
            if param.grad is not None:
                param.grad = param.grad.to(dist_util.dev())        

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            for group in self.opt.param_groups:
                for param in group['params']:
                    if param is not None:
                        param.data = param.data.to(dist_util.dev())
                        if param.grad is not None:
                            param.grad = param.grad.to(dist_util.dev())
  
    def run_loop(self):
        dataloader = iter(self.data)
        dinov2_model = DINOv2WithHiddenStates(self.image_encoder)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):       
            try:
                batch = next(dataloader)
            except StopIteration:     
                dataloader = iter(self.data)
                batch = next(dataloader)
            
            cond_keys = ['agn', 'agn_mask', 'image_densepose', 'vit_cloth', 'vit_cloth_sr']
            cond = {key: batch[key].to(dist_util.dev()) if isinstance(batch[key], th.Tensor) else batch[key] for key in cond_keys}

            with th.no_grad():
                target = self.get_first_stage_encoding(batch['image'].to(dist_util.dev())) 
                cond['agn'] = self.get_first_stage_encoding(cond['agn'].to(dist_util.dev()))
                cond['agn_mask'] = self.get_first_stage_encoding(batch['agn_mask'].to(dist_util.dev()))
                cond['image_densepose'] = self.get_first_stage_encoding(batch['image_densepose'].to(dist_util.dev()))

                with autocast(dtype=th.bfloat16):
                    cond['vit_cloth'] = th.stack(dinov2_model(cond['vit_cloth']), dim=1) # is [B, 40, 257, 1536]
                    cond['vit_cloth_sr'] = th.stack(dinov2_model(cond['vit_cloth_sr']), dim=1)

            self.run_step(target, cond)

            if self.step % self.log_interval == 0:
                out = logger.dumpkvs()
                if logger.get_rank_without_mpi_import()==0:
                    print(f"[Step {self.resume_step+self.step}] loss: {out.get('loss'):.4f}, mse: {out.get('mse'):.4f}")
            if self.step % self.save_interval == 0:
                self.opt.consolidate_state_dict()
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, target, cond):
        self.forward_backward(target, cond)
        took_step = self.mp_trainer.optimize(self.opt)

        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step() 

    def forward_backward(self, target, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, target.shape[0], self.microbatch):
            micro = target[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {}
            
            micro_cond['agn'] = cond['agn'][i : i + self.microbatch].to(dist_util.dev())
            micro_cond['agn_mask'] = cond['agn_mask'][i : i + self.microbatch].to(dist_util.dev())
            micro_cond['image_densepose'] = cond['image_densepose'][i : i + self.microbatch].to(dist_util.dev())
            micro_cond['vit_cloth'] = cond['vit_cloth'][i : i + self.microbatch].to(dist_util.dev())
            micro_cond['vit_cloth_sr'] = cond['vit_cloth_sr'][i : i + self.microbatch].to(dist_util.dev())
            
            last_batch = (i + self.microbatch) >= target.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs={"cond": micro_cond},
            )

            compute_losses_mask = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs={"cond": micro_cond, "enable_mask": True},
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
                losses_mask = compute_losses_mask()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
                    losses_mask = compute_losses_mask()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach() + losses_mask["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean() + (losses_mask["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            log_loss_dict(
                self.diffusion, t, {'m_'+k: v * weights for k, v in losses_mask.items()}
            ) 
            self.mp_trainer.backward(loss)


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


