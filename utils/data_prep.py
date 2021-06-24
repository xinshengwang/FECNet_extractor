from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class FECData(Dataset):
    """Google face comparing dataset."""

    def __init__(self,args):
        self.args = args
        self.filenames = self.load_filenames(args.data_root)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)])
    
    def load_filenames(self,root):
        video_names = os.listdir(root)
        image_paths= []
        for video in video_names:
            video_path = os.path.join(root,video)
            names = os.listdir(video_path)
            for name in names:
                path = os.path.join(video_path,name)
                image_paths.append(path)
        return image_paths

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.norm(img)
        return img, img_path