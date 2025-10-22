import os
from os.path import join as opj
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

def safe_imread(p: str):
    base, _ = os.path.splitext(p)
    if os.path.exists(p):
        return cv2.imread(p)

    for ext in ('.png', '.jpeg', '.jpg'):  
        alt_path = base + ext
        if alt_path == p:
            continue
        if os.path.exists(alt_path):
            return cv2.imread(alt_path)
    return None

def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None,
):
    if img is None:
        img = safe_imread(p)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from path or alternatives: {p}")

    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = safe_imread(p)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from path or alternatives: {p}")

    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img

def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            vit_img_H,
            vit_img_W,
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
            transform_size=None, 
            transform_color=None,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.vit_img_H = vit_img_H
        self.vit_img_W = vit_img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0

        self.resize_transform = A.Resize(img_H, img_W)
        
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None
        self.transform_vit_size = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]

        if transform_size is not None:
            if "hflip" in transform_size:
                transform_size_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "image_densepose":"image", 
                                    }
        )

        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image",
                                    "image_densepose":"image", 
                                    "vit_cloth":"image",
                                    "vit_cloth_sr":"image",
                                    },
                is_check_shapes=False
        )

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image",
                                    "vit_cloth_sr":"image",
                                    "vit_cloth":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]

        if self.transform_size is None and self.transform_color is None:
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                in_inverse_mask=True
            )
            vit_cloth_sr = imread(
                opj(self.drd, self.data_type, "cloth_sr", self.c_names[self.pair_key][idx]), 
                self.vit_img_H, 
                self.vit_img_W
            )
            vit_cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
                self.vit_img_H, 
                self.vit_img_W
            )

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)

        else:
            agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), is_mask=True)
            vit_cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            vit_cloth_sr = imread_for_albu(opj(self.drd, self.data_type, "cloth_sr", self.c_names[self.pair_key][idx]))
            image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))

            if self.transform_size is not None:
                transformed = self.transform_size(
                    image=image, 
                    agn=agn, 
                    agn_mask=agn_mask, 
                    image_densepose=image_densepose,
                    vit_cloth=vit_cloth, 
                    vit_cloth_sr=vit_cloth_sr,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                vit_cloth_sr=transformed["vit_cloth_sr"]
                vit_cloth=transformed["vit_cloth"]

             
            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                image_densepose=transformed_image["image_densepose"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    vit_cloth=vit_cloth,
                    vit_cloth_sr=vit_cloth_sr,
                )

                cloth=transformed_cloth["image"]
                vit_cloth=transformed_cloth["vit_cloth"]
                vit_cloth_sr=transformed_cloth["vit_cloth_sr"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn, 
                    vit_cloth=vit_cloth,
                    vit_cloth_sr=vit_cloth_sr,
                )


                image=transformed["image"]
                agn=transformed["agn"]
                vit_cloth=transformed["vit_cloth"]
                vit_cloth_sr=transformed["vit_cloth_sr"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)

            vit_cloth=cv2.resize(vit_cloth, (self.vit_img_H, self.vit_img_W))
            vit_cloth_sr=cv2.resize(vit_cloth_sr, (self.vit_img_H, self.vit_img_W))

            agn = norm_for_albu(agn)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            vit_cloth = norm_for_albu(vit_cloth)
            vit_cloth_sr = norm_for_albu(vit_cloth_sr)
            agn_mask = np.repeat(agn_mask, 3, axis=2) # to make channel size into 3 from 1. [w, h, 3]

        return dict(
            agn=agn.transpose((2,0,1)), 
            agn_mask=agn_mask.transpose((2,0,1)),
            image=image.transpose((2,0,1)),
            image_densepose=image_densepose.transpose((2,0,1)),
            vit_cloth=vit_cloth.transpose((2,0,1)),
            vit_cloth_sr=vit_cloth_sr.transpose((2,0,1)),
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )
    