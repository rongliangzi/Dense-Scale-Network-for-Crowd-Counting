import torch
import torchvision
import torch.nn as nn
import os
import glob
from modeling.dsnet import DenseScaleNet as DSNet
import torchvision.transforms as transforms
from dataset import RawDataset
import torch.nn.functional as F
import logging
import warnings
warnings.filterwarnings("ignore")


def cal_lc_loss(output, target, sizes=(1,2,4)):
    criterion_L1 = nn.L1Loss(reduction='sum')
    Lc_loss = None
    for s in sizes:
        pool = nn.AdaptiveAvgPool2d(s)
        est = pool(output)
        gt = pool(target)
        if Lc_loss:
            Lc_loss += criterion_L1(est, gt) / s**2
        else:
            Lc_loss = criterion_L1(est, gt) / s**2
    return Lc_loss


def getLogger(filename):
    logger = logging.getLogger('train_logger')

    while logger.handlers:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s], ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def val(model, test_loader):
    model.eval()
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for img, target, count in test_loader:
            img = img.cuda()
            output = model(img)
            est_count = output.sum().item()
            mae += abs(est_count - count)
            mse += (est_count - count)**2
    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse**0.5
    return float(mae), float(mse)

def get_loader(train_path, test_path, ratio, kernel_path):
    train_img_paths = []
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        train_img_paths.append(img_path)
    test_img_paths = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_img_paths.append(img_path)
    # ! ToTensor -> range(0,1)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_loader = torch.utils.data.DataLoader(
        RawDataset(train_img_paths, transform, aug=True, ratio=ratio, kernel_path=kernel_path),
        shuffle=True, batch_size=1, num_workers=4)
    # ratio 1 for val, gt_map is not used
    test_loader = torch.utils.data.DataLoader(
        RawDataset(test_img_paths, transform, ratio=1, aug=False, kernel_path=kernel_path), 
        shuffle=False, batch_size=1, num_workers=4)
    return train_loader, test_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    net = DSNet('')
    print('%s trainable params'%count_parameters(net))
    net.cuda()



     # For shg A
    train_path = '...'
    test_path =  '...'
    kernel_path = 'maps_adaptive_kernel'
    lbda = 1000
    
    ratio = 8 # density map scaling
    
    train_loader, test_loader = get_loader(train_path, test_path, ratio, kernel_path)
    save_path = './saved_models/dsnet_shtechA.pth'


    epochs = 500
    lr=1e-6
    weight_decay=5e-4
    
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    #best_mae, _  = val(net, test_loader)
    best_mae = 167.99
    logger = getLogger('logs/dsnet_sha_Adam_1e-6.txt')
    for epoch in range(epochs):
        train_loss = 0.0
        net.train()
        for img, target, count in train_loader:
            optimizer.zero_grad()
            img = img.cuda()
            target = target.unsqueeze(1).cuda()
            output = net(img)
            Le_Loss = criterion(output, target)
            Lc_Loss = cal_lc_loss(output, target)
            loss = Le_Loss + lbda * Lc_Loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        mae, mse = val(net, test_loader)
        
        logger.info('Epoch {}/{} Loss:{:.3f}, MAE:{:.2f}, MSE:{:.2f}, Best MAE:{:.2f}'.format(epoch+1, epochs, train_loss/len(train_loader), mae, mse, best_mae))
        if mae < best_mae:
            best_mae = mae
            torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
    