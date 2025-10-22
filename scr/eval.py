import argparse
import json
import os
from typing import List, Tuple, Dict

import PIL.Image
import torch
from cleanfid import fid
from torch.utils.data import DataLoader 
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm

from generate_fid_stats import make_custom_stats

class GTTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot: str, dataset: str, category: str, transform: transforms.Compose):
        assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
        assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

        self.dataset = dataset
        self.category = category
        self.transform = transform
        self.dataroot = dataroot

        if dataset == 'dresscode':
            filepath = os.path.join(dataroot, f"test_pairs_paired.txt")
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()

            if category in ['lower_body', 'upper_body', 'dresses']:
                self.paths = sorted(
                    [os.path.join(dataroot, category, 'images', line.strip().split()[0]) for line in lines if
                     os.path.exists(os.path.join(dataroot, category, 'images', line.strip().split()[0]))])
            else:
                self.paths = sorted(
                    [os.path.join(dataroot, category, 'images', line.strip().split()[0]) for line in lines for
                     category in ['lower_body', 'upper_body', 'dresses'] if
                     os.path.exists(os.path.join(dataroot, category, 'images', line.strip().split()[0]))])
        else:  # vitonhd
            filepath = os.path.join(dataroot, f"test_pairs.txt")
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()
            self.paths = sorted([os.path.join(dataroot, 'test', 'image', line.strip().split()[0]) for line in lines])

        # Debug print
        print(f"GTTestDataset initialized with {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name

class GenTestDataset(torch.utils.data.Dataset):
    def __init__(self, gen_folder: str, dataset: str, category: str, transform: transforms.Compose):
        assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
        assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

        self.dataset = dataset
        self.category = category
        self.transform = transform
        self.gen_folder = gen_folder

        if dataset == 'dresscode':
            if category in ['lower_body', 'upper_body', 'dresses']:
                self.paths = sorted(
                    [os.path.join(gen_folder, category, name) for name in os.listdir(os.path.join(gen_folder, category))]
                )
            elif category == 'all':
                existing_categories = []
                for cat in ['lower_body', 'upper_body', 'dresses']:
                    if os.path.exists(os.path.join(gen_folder, cat)):
                        existing_categories.append(cat)

                self.paths = sorted(
                    [os.path.join(gen_folder, cat, name) for cat in existing_categories for
                     name in os.listdir(os.path.join(gen_folder, cat)) if
                     os.path.exists(os.path.join(gen_folder, cat, name))]
                )
            else:
                raise ValueError('Unsupported category')
        
        elif dataset == 'vitonhd':
            # vitonhd does not have sub-categories, so we handle it differently
            self.paths = sorted(
                [os.path.join(gen_folder, name) for name in os.listdir(gen_folder) if
                 name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            )

        # Debug print to confirm the number of images found
        print(f"GenTestDataset initialized with {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name

def compute_metrics(gen_folder: str, test_order: str, dataset: str, category: str, metrics2compute: List[str],
                    dresscode_dataroot: str, vitonhd_dataroot: str, generated_size: Tuple[int, int] = (512, 512),
                    batch_size: int = 16, workers: int = 8) -> Dict[str, float]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert test_order in ['paired', 'unpaired']
    assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
    assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

    if dataset == 'dresscode':
        gt_folder = dresscode_dataroot
    elif dataset == 'vitonhd':
        gt_folder = vitonhd_dataroot
    else:
        raise ValueError('Unsupported dataset')


    for m in metrics2compute:
        assert m in ['all', 'ssim_score', 'lpips_score', 'fid_score', 'kid_score'], 'Unsupported metric'

    if metrics2compute == ['all']:
        metrics2compute = ['ssim_score', 'lpips_score', 'fid_score', 'kid_score']


    if category == 'all':
        if "fid_score" in metrics2compute or "all" in metrics2compute:
            if not fid.test_stats_exists(f"{dataset}_all", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            fid_score = fid.compute_fid(gen_folder, dataset_name=f"{dataset}_all", mode='clean', dataset_split="custom", verbose=True, use_dataparallel=False)
        if "kid_score" in metrics2compute or "all" in metrics2compute:
            if not fid.test_stats_exists(f"{dataset}_all", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            kid_score = fid.compute_kid(gen_folder, dataset_name=f"{dataset}_all", mode='clean', dataset_split="custom", verbose=True, use_dataparallel=False)
    else:
        if "fid_score" in metrics2compute or "all" in metrics2compute:
            if not fid.test_stats_exists(f"{dataset}_{category}", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            fid_score = fid.compute_fid(os.path.join(gen_folder, category), dataset_name=f"{dataset}_{category}", mode='clean', verbose=True, dataset_split="custom", use_dataparallel=False)
        if "kid_score" in metrics2compute or "all" in metrics2compute:
            if not fid.test_stats_exists(f"{dataset}_{category}", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            kid_score = fid.compute_kid(os.path.join(gen_folder, category), dataset_name=f"{dataset}_{category}", mode='clean', verbose=True, dataset_split="custom", use_dataparallel=False)
            
    trans = transforms.Compose([
        transforms.Resize(generated_size),
        transforms.ToTensor(),
    ])
    print("gen_folder path:", gen_folder)
    print("gt_folder path:", gt_folder)
    gen_dataset = GenTestDataset(gen_folder, dataset, category, transform=trans)
    gt_dataset = GTTestDataset(gt_folder, dataset, category, trans)

    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Debug print for data loader size
    print(f"Total batches in gen_loader: {len(gen_loader)}")
    print(f"Total batches in gt_loader: {len(gt_loader)}")
    
    for idx, (gen_batch, gt_batch) in tqdm(enumerate(zip(gen_loader, gt_loader)), total=len(gt_loader)):
        gen_images, gen_names = gen_batch
        gt_images, gt_names = gt_batch
        
        if len(gen_images) == 0 or len(gt_images) == 0:
            print(f"Empty batch at index {idx}")
            continue

        assert gen_names == gt_names, f"Names do not match at index {idx}"

        gen_images = gen_images.to(device)
        gt_images = gt_images.to(device)

        if "ssim_score" in metrics2compute or "all" in metrics2compute:
            ssim.update(gen_images, gt_images)

        if "lpips_score" in metrics2compute or "all" in metrics2compute:
            lpips.update(gen_images, gt_images)
    
    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        ssim_score = ssim.compute()
    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        lpips_score = lpips.compute()

    results = {}

    for m in metrics2compute:
        if torch.is_tensor(locals()[m]):
            results[m] = locals()[m].item()
        else:
            results[m] = locals()[m]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument("--gen_folder", type=str, help="Path to the generated images")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
    parser.add_argument("--test_order", type=str, choices=['paired', 'unpaired'])
    parser.add_argument("--dataset", type=str, default='', choices=['dresscode', 'vitonhd'],
                        help="Dataset to use for the metrics")
    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the dataloaders")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for the dataloaders")

    args = parser.parse_args()

    if args.dataset == "vitonhd":
        args.category = 'all'  # vitonhd does not have sub-categories

    if not os.path.exists(args.gen_folder):
        raise ValueError("The generated images folder does not exist")

    metrics = compute_metrics(args.gen_folder, args.test_order, args.dataset, args.category, ['all'],
                              args.dresscode_dataroot, args.vitonhd_dataroot, batch_size=args.batch_size,
                              workers=args.workers)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    with open(os.path.join(args.gen_folder, f"metrics_{args.test_order}_{args.category}.json"), "w+") as f:
        json.dump(metrics, f, indent=4)
