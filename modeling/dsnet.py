import torch
import torch.nn as nn
import torchvision.models as models

# Forgot to add 1x1 conv layer after dilated layer:
# nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
# Forgot to put bias to False in dilated layer
# No RELu after last conv 3x3
# Each dilated layer within the block is densely connected with others as in DenseASPP so that each layer can access to all the subsequent layers 
class DDCB(nn.Module):
    def __init__(self, in_planes):
        super(DDCB, self).__init__()
        # add a 1 × 1 convolutional layer before every dilated layer to prevent the block from growing too wide.
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, 256, 1,bias=False), nn.ReLU(True), nn.Conv2d(256, 64, 3, padding=1,bias=False), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes+64, 256, 1,bias=False), nn.ReLU(True), nn.Conv2d(256, 64, 3, padding=2, dilation=2,bias=False), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes+64*2, 256, 1,bias=False), nn.ReLU(True), nn.Conv2d(256, 64, 3, padding=3, dilation=3,bias=False), nn.ReLU(True))
        # 3 × 3 filter size is adopted to fuse concated features from front layers and reduce channel number in the end.
        self.conv4 = nn.Conv2d(in_planes+64*3, 512, 3,bias=False, padding=1)
    def forward(self, x):
        #print(x.shape)
        x1_raw = self.conv1(x)
        #print(x1_raw.shape)
        x1 = torch.cat([x, x1_raw], 1)
        #print(x1.shape)
        x2_raw = self.conv2(x1)
        #print(x2_raw.shape)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        #print(x2.shape)
        x3_raw = self.conv3(x2)
        #print(x3_raw.shape)
        x3 = torch.cat([x, x1_raw, x2_raw, x3_raw], 1)
        #print(x3.shape)
        output = self.conv4(x3)
        return output

 
class DenseScaleNet(nn.Module):
    def __init__(self, load_model=''):
        super(DenseScaleNet, self).__init__()
        self.load_model = load_model
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg)
        # ?! Freeze vgg-18 params
        for param in self.features.parameters():
            param.requires_grad = False
        self.DDCB1 = DDCB(512)
        self.DDCB2 = DDCB(512)
        self.DDCB3 = DDCB(512)
        self.output_layers = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(True), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x1_raw = self.DDCB1(x)
        x1 = x1_raw + x
        x2_raw = self.DDCB2(x1)
        x2 = x2_raw + x1_raw + x
        x3_raw = self.DDCB3(x2)
        x3 = x3_raw + x2_raw + x1_raw + x
        output = self.output_layers(x3)
        return output
    def _initialize_weights(self):
        self_dict = self.state_dict()
        pretrained_dict = dict()
        self._random_initialize_weights()
        if not self.load_model:
            vgg16 = models.vgg16(pretrained=True)
            print('vgg imported')
            for k, v in vgg16.named_parameters(): 
                #print(k)
                #print(v)
                if k in self_dict and self_dict[k].size() == v.size():
                    print('layer %s is updated with pretrained vgg-16'%k)
                    pretrained_dict[k] = v
            self_dict.update(pretrained_dict)
            self.load_state_dict(self_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))
            
    def _random_initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                #nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)