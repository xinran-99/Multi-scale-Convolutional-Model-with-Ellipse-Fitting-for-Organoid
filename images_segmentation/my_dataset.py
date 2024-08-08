import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor

mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])
class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "validation"
        data_root = os.path.join(root, "OriginalData", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.mask = [os.path.join(data_root, "segmentations", i) for i in img_names]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # check files
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exist.")
    def apply_clahe(self, img):
        # Apply CLAHE to the input image
        img_array = np.array(img)
        img_array_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        img_array_lab[:, :, 0] = self.clahe.apply(img_array_lab[:, :, 0])
        img_clahe = cv2.cvtColor(img_array_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask.astype(np.float32))

        # # Convert ground truth to binary mask
        # mask = np.where(manual > 0.5, 1, 0)

        # # Convert mask to PIL Image
        # mask = Image.fromarray(mask.astype(np.uint8) * 255)
        img = self.apply_clahe(img)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)


