import torch
import torch.nn as nn
import copy

def MTL(class_num, pretrained=False, path=None, **kwargs):
    model = MTLResNet(Bottleneck, [2,2,2], class_num, **kwargs)
    return model

class Reshape(nn.Module):
    def __init__(self,*args):
        super(Reshape,self).__init__()
        self.shape = args
    
    def forward(self,x):
        return x.view(x.size(0),-1)


class MTLResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, groups=1, width_per_group=64):
        super(MTLResNet, self).__init__()
        
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        self.sharedlayer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True), # batchsize, 16, 28, 28
            self._make_layer(block, 16, layers[0]),
            self._make_layer(block, 32, layers[1], stride=2),
            self._make_layer(block, 64, layers[2], stride=2)# batchsize,256,7,7
           
        )

        self.classify = nn.Sequential(   
            nn.AdaptiveAvgPool2d((1, 1)) ,  
            Reshape(),
            nn.Linear(64 * block.expansion, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) ,  
            Reshape(),
            # nn.AvgPool2d(2,stride=2), # batchsize, 16 ,14 ,14
            # Reshape(),
            # nn.Linear(16*14*14, 1024),
            nn.Linear(64 * block.expansion,1),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(32, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sharedlayer(x)
        # x = x.view(x.size(0), -1) 
        # print(x.size())
        out_c = self.classify(x)
        out_d = self.discriminator(x)
        return out_c,out_d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv3x3(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)