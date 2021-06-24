### imports
from concurrent import futures
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from functools import partial
from models.FECNet import FECNet
from utils.data_prep import FECData
from concurrent.futures import ProcessPoolExecutor
Executor = ProcessPoolExecutor(max_workers=20)

def save_feature(features,paths,args):
    futures = []
    for i,path in enumerate(paths):
        feature = features[i]
        video_path,image_name = os.path.split(path)
        save_root = video_path.replace('images_crop','expression')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root,image_name.replace('.jpg','.npy'))
        futures.append(Executor.submit(partial(np.save,save_path,feature)))
    for future in futures:
        future.result()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch FECNet')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--data_root',type=str,default=r'F:\dataset\Avatar\Obama\clip\images_crop')
    parser.add_argument('--save_root',type=str,default=r'F:\dataset\TTS_TEST\expression')
    parser.add_argument('--model_dir',type=str,default=r'F:\code\Face\FECNet.pt')
    parser.add_argument('--batch_size', type=int, default=240,
                        help='input batch size for training (default: 240)')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.',default=0)
    parser.add_argument('--pretrained', dest='pretrained', type=bool,
                        help='Use pretrained weightts of FECNet.', default=True)
    args = parser.parse_args()

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")

    data = FECData(args)
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)
    model = FECNet(pretrained=args.pretrained,model_dir=args.model_dir).cuda()
    for image,paths in tqdm(dataloader):
        image = image.float().cuda()
        model = model.eval().cuda()
        with torch.no_grad():
            features = model(image)
        features = features.detach().cpu().numpy()
        save_feature(features,paths,args)

