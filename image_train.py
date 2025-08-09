import argparse

from masked_diffusion import dist_util, logger
from masked_diffusion.resample import create_named_schedule_sampler
from masked_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from masked_diffusion.train_util import TrainLoop
from masked_diffusion import create_diffusion, model_and_diffusion_defaults, diffusion_defaults
import masked_diffusion.models as models_mdt
import os

from importlib import import_module
from torch.utils.data import DataLoader, ConcatDataset

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist_multinode(args)
    logger.configure(format_strs=['tensorboard', 'csv'])
    logger.log("creating model and diffusion...")
    configs = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(configs)
    print(args)
 
    image_size = configs['image_size']
    latent_size = image_size // 8

    model = models_mdt.__dict__[args.model](input_size=latent_size, mask_ratio=args.mask_ratio, decode_layer=args.decode_layer).to(dist_util.dev())
    print(model)

    diffusion = create_diffusion(**args_to_dict(args, diffusion_defaults().keys())) #.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_dataset = []

    dataset = getattr(import_module("scr.dataset_vitonhd"), 'VITONHDDataset')(
        data_root_dir=os.path.join(args.data_dir, 'zalando-hd-resized'), 
        img_H=args.image_size, 
        img_W=args.image_size, 
        vit_img_H=args.vit_image_size,
        vit_img_W=args.vit_image_size,
        transform_size=args.transform_size, 
        transform_color=args.transform_color, 
    )
    train_dataset.append(dataset)
    
    dc_data = ['DressCode/dresses', 'DressCode/lower_body', 'DressCode/upper_body']
    for d_path in dc_data:
        dataset = getattr(import_module("scr.dataset_dresscode"), 'DressCodeDataset')(
            data_root_dir=os.path.join(args.data_dir, d_path), 
            img_H=args.image_size, 
            img_W=args.image_size, 
            vit_img_H=args.vit_image_size,
            vit_img_W=args.vit_image_size,
            transform_size=args.transform_size, 
            transform_color=args.transform_color, 
        )
        train_dataset.append(dataset)
    train_dataset = ConcatDataset(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=12,
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=True, 
        pin_memory=True
    )
    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000, 
        resume_checkpoint="", # path/to/model.pt
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model="MDT_IVTON",
        mask_ratio=None,
        image_size=512,
        vit_image_size=224,
        decode_layer=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument(
        "--rank", default=0, type=int, help="""rank for distrbuted training."""
    )
    parser.add_argument(
        "--work_dir", type=str, required=True, default='experiments', help="Path to saving the expriment"
    )

    #for dataset
    parser.add_argument('--transform_size', default='shiftscale hflip')
    parser.add_argument('--transform_color', default='hsv bright_contrast')
    parser.add_argument('--n_gpus', default=1, type=int)

    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":

    main()
