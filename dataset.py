from torch.utils.data import Dataset
from PIL import Image
import h5py
import numpy as np
import cv2
import random

from skimage import exposure, img_as_float


def load_data(img_path, ratio, aug, index, kernel_path='maps_adaptive_kernel'):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', kernel_path)
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if aug:
        crop_size = (img.size[0]//2, img.size[1]//2)
        
        if random.random() <=0.44:
            # 4 non-overlapping patches
            dx = int(random.randint(0,1) * crop_size[0])
            dy = int(random.randint(0,1) * crop_size[1])
        else:
            # 5 random patches
            # set seed to ensure for each image the random patches are certain
            # if not set, the crop will be online which means the patches change every time loading, leading to a dynamic training set.
            patch_id = random.randint(0, 4)
            random.seed(index + patch_id * 0.1)
            dx = int(random.random() * crop_size[0])
            random.seed(index + 0.5 + patch_id * 0.1)
            dy = int(random.random() * crop_size[1])
        # crop
        img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
        # flip
        if random.random() > 0.5:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # gamma transform
        if random.random() > 0.7:
            image = img_as_float(img)
            # gamma_img: np.array(dtype=float64) ranging [0,1]
            if random.random() > 0.5:
                gamma_img = exposure.adjust_gamma(image, 1.5)
            else:
                gamma_img = exposure.adjust_gamma(image, 0.5)
            gamma_img = gamma_img * 255
            gamma_img = np.uint8(gamma_img)
            img = Image.fromarray(gamma_img)
        # grayscale 
        if random.random() > 0.9:
            img = img.convert('L').convert('RGB') # convert to grayscale on 3 channels
    count = target.sum()
    if ratio>1:
        target = cv2.resize(target, (int(target.shape[1]/ratio),int(target.shape[0]/ratio)), interpolation=cv2.INTER_CUBIC) * (ratio**2)
    
    return img, target, count


class RawDataset(Dataset):
    def __init__(self, root, transform, ratio=8, aug=False, kernel_path='maps_adaptive_kernel'):
        self.nsamples = len(root)
        self.aug = aug
        self.root = root
        self.ratio = ratio
        self.transform = transform
        self.kernel_path = kernel_path
    def __getitem__(self, index):
        img, target, count = load_data(self.root[index], self.ratio, self.aug, index, self.kernel_path)
        if self.transform:
            img = self.transform(img)
        return img, target, count
    def __len__(self):
        return self.nsamples