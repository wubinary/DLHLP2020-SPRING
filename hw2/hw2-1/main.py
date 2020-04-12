from dataset import VCTK_Dataset, DataLoader
from train import train

import os, warnings, argparse

def parse_args(string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4,
                        type=float, help='leanring rate')
    parser.add_argument('--epoch', default=8,
                        type=int, help='epochs')
    parser.add_argument('--batch-size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--gpu', default="0",
                        type=str, help='0:1080ti 1:1070')
    parser.add_argument('--num-workers', default=6,
                        type=int, help='dataloader num workers')
    parser.add_argument('--save-path', default='trained_model',
                        type=str, help='.pth model file save dir')
    
    args = parser.parse_args() if string is None else parser.parse_args(string)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    return args
    
if __name__=='__main__':
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu #0:1080ti 1:1070
    warnings.filterwarnings("ignore")
    
    ## load dataset
    train_dataset = VCTK_Dataset('preprocess/vctk.h5', 'preprocess/sample_segments/train_samples', seg_len=128, mode='train')
    valid_dataset = VCTK_Dataset('preprocess/vctk.h5', 'preprocess/sample_segments/valid_samples', seg_len=128, mode='test')

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size,
                                  #num_workers = args.num_workers,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=args.batch_size*4,)
                                  #num_workers = args.num_workers)
    
    ## train
    train(args, train_dataloader, valid_dataloader)
