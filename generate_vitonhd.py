import torch
from torchvision.utils import save_image
from masked_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import os
from torch.cuda.amp import autocast
from importlib import import_module
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

# argparse argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--image_size', type=int)
parser.add_argument('--vit_img_size', type=int)
parser.add_argument('--num_sampling_steps', type=int)
parser.add_argument('--cfg_scale', type=float)
parser.add_argument('--pow_scale', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--unpair', type=str, choices=['true', 'false'],
                    help="Must be 'true' or 'false'.")
args = parser.parse_args() 
args.unpair = args.unpair.lower() == 'true'  # Convert string to boolean

# assign to variables
data_dir = args.data_dir
output_dir = args.output_dir
model_path = args.model_path
image_size = args.image_size
vit_img_size = args.vit_img_size
num_sampling_steps = args.num_sampling_steps
cfg_scale = args.cfg_scale
pow_scale = args.pow_scale
batch_size = args.batch_size
unpair = args.unpair

class VAE:
    def __init__(self, vae, device=None):
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor
        self.device = device

    def encode(self, x):
        z = self.vae.encode(x).latent_dist.sample()
        z = z * self.scaling_factor
        if self.device:
            z = z.to(self.device)
        return z

    def decode(self, z):
        z = z / self.scaling_factor
        if self.device:
            z = z.to(self.device)
        return self.vae.decode(z).sample

class DINOv2WithHiddenStates(torch.nn.Module):
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

# Function to shrink the mask by eroding it
def shrink_mask(mask, erosion_width=4):
    # Erode the mask by applying max pooling with a kernel size equal to the erosion width
    eroded_mask = F.max_pool2d(1 - mask.unsqueeze(0).float(), kernel_size=erosion_width, stride=1, padding=erosion_width // 2).squeeze(0)
    eroded_mask = 1 - eroded_mask  # Revert the mask to original values
    return eroded_mask.clamp(0, 1)  # Ensure the values remain in [0, 1]

# Setup PyTorch:
torch.manual_seed(0)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

from masked_diffusion.models import MDT_IVTON_XL

img_H = image_size
img_W = image_size
latent_size = image_size // 8
vit_img_H = vit_img_size
vit_img_W = vit_img_size

parent, filename = os.path.split(model_path) 
model_name, _ = os.path.splitext(filename) 
_, parent_folder = os.path.split(parent)

if unpair:
    save_dir = os.path.join(output_dir, f"{parent_folder}_{model_name}", 'VITON-HD', 'unpair')
else:
    save_dir = os.path.join(output_dir, f"{parent_folder}_{model_name}", 'VITON-HD', 'pair')

model = MDT_IVTON_XL(input_size=latent_size, decode_layer=4).to(device)

state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
diffusion = create_diffusion(str(num_sampling_steps))

vae_ = AutoencoderKL.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",subfolder="vae").to(device)
vae = VAE(vae_, device=device)

### start PXT
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
dinov2_vitg14.eval()
for param in dinov2_vitg14.parameters():
    param.requires_grad = False
img_encoder = DINOv2WithHiddenStates(dinov2_vitg14)

num_samples = batch_size

dataset = getattr(import_module("scr.dataset_vitonhd"), 'VITONHDDataset')(
    data_root_dir=os.path.join(data_dir, 'zalando-hd-resized'), 
    img_H=img_H,
    img_W=img_W,
    vit_img_H=vit_img_H,
    vit_img_W=vit_img_W,
    is_paired=not unpair,
    is_test=True,
    is_sorted=True
)

dataloader = DataLoader(
    dataset,
    num_workers=4, 
    batch_size=max(batch_size, 1), 
    shuffle=False, 
    pin_memory=True
)

shape = (4, img_H//8, img_W//8)  # [4,64,48]

cond_keys = ['agn', 'agn_mask', 'image_densepose', 'vit_cloth', 'vit_cloth_sr', 'img_fn', 'cloth_fn']

total_iter = len(dataloader)
total_image = len(dataset)
generated_image_count = 0 
for i, batch in enumerate(dataloader):
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    cond = {key: batch[key] for key in cond_keys}
    
    batch_size_now = next(iter(batch.values())).size(0) 
    generated_image_count += batch_size_now

    print(f'Iteration {i}/{total_iter}, {generated_image_count}/{total_image} images.')

    repaint_agn = cond['agn']
    repaint_agn_mask = cond['agn_mask']

    if len(batch['cloth_fn']) != num_samples:
        num_samples = len(batch['cloth_fn'])

    with torch.no_grad():
        cond['agn'] = vae.encode(cond['agn'])
        cond['image_densepose'] = vae.encode(cond['image_densepose'])
        cond['agn_mask'] = vae.encode(cond['agn_mask'])
        with autocast(dtype=torch.bfloat16):
            cond['vit_cloth'] = torch.stack(img_encoder(cond['vit_cloth']), dim=1) # is [B, 40, 257, 1536]
            cond['vit_cloth_sr'] = torch.stack(img_encoder(cond['vit_cloth_sr']), dim=1)

    # Create sampling noise
    z = torch.randn(num_samples, 4, latent_size, latent_size, device=device)

    model_kwargs = dict(cfg_scale=cfg_scale, scale_pow=pow_scale, 
                        cond=cond
                        )

    ### use DDIM, PXT modified
    # DDIM solver
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples = vae.decode(samples)

    # Simple postprocessing to use original agnostic image and fill in the generated inpainting.
    if True: 
        repainted_samples = []
        for i, (sample_img, agn_img, mask_img) in enumerate(zip(samples, repaint_agn, repaint_agn_mask)):
            sample_img = sample_img.cpu().clone()
            agn_img = agn_img.cpu().clone()
            mask_img = mask_img.cpu().clone()

            # Resize the mask if necessary to match the sample size
            if mask_img.shape != sample_img.shape:
                mask_img = F.interpolate(mask_img.unsqueeze(0), size=sample_img.shape[1:], mode='nearest').squeeze(0)

            # Shrink the mask to exclude boundary pixels
            smaller_mask = shrink_mask(mask_img, erosion_width=4)

            # Resize smaller_mask to match sample_img size to avoid dimension mismatch
            if smaller_mask.shape != sample_img.shape:
                smaller_mask = F.interpolate(smaller_mask.unsqueeze(0), size=sample_img.shape[1:], mode='nearest').squeeze(0)

            # Blend the images using the smaller mask
            repainted_img = agn_img * smaller_mask + sample_img * (1 - smaller_mask)

            # Append the repainted image to the output list
            repainted_samples.append(repainted_img)

        repainted_samples = torch.stack(repainted_samples).to(samples.device)
        samples = repainted_samples

    # Save 
    os.makedirs(save_dir, exist_ok=True)
    for i, img_fn in enumerate(batch['img_fn']):
        new_filename = img_fn
        new_image_name = os.path.join(save_dir, new_filename)
        save_image(samples[i].squeeze(), new_image_name, normalize=True, value_range=(-1, 1))
