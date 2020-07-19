import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
mpl.use('Agg')
import argparse
from modeling.dsnet import DenseScaleNet as DSNet
import torch
from dataset import RawDataset
import torchvision.transforms as transforms
import os
import glob
from utils.functions import *
import matplotlib.pyplot as plt
from matplotlib import cm as CM


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_loader(args):
    test_img_paths = []
    for img_path in glob.glob(os.path.join(args.test_img_dir, '*.jpg')):
        test_img_paths.append(img_path)
    test_loader = torch.utils.data.DataLoader(RawDataset(test_img_paths, transform, ratio=1, aug=False), shuffle=False, batch_size=1)
    return test_loader, test_img_paths
    

def val(model, test_loader):
    print('validation!')
    model.eval()
    mae, rmse = 0.0, 0.0
    with torch.no_grad():
        for it,data in enumerate(test_loader):
            img, target, count = data[0:3]
            img = img.cuda()
            dmp = model(img)
            est_count = dmp.sum().item()
            mae += abs(est_count - count.item())
            rmse += (est_count - count.item())**2
            print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
    mae /= len(test_loader)
    rmse /= len(test_loader)
    rmse = rmse**0.5
    return mae, rmse


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    net = DSNet(args.model_path)
    net.load_state_dict(torch.load(args.model_path))
    print('{} loaded!'.format(args.model_path))
    net.cuda()
    test_loader, test_img_paths = get_loader(args, imgs)
    mae, rmse = val(net, test_loader)
    print('{} MAE:{:.2f}, RMSE:{:.2f}'.format(args.model_path, mae, rmse))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense Scale Net Dataset Test')
    parser.add_argument('--gpu', default='0', help='assign device')
    parser.add_argument('--model_path', metavar='model path', type=str)
    parser.add_argument('--test_img_dir', metavar='test image directory', type=str)
    
    args = parser.parse_args()
    main(args)