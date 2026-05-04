import torch.nn as nn
from src.resnet_block import ResidualBlock
class MiniResNet(nn.Module):
    
    def __init__(self,num_classes=10):
        super(MiniResNet, self).__init__()
        #1. Initial layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,stride=1,bias=False)
        
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        
        
        #2. Residual Blocks
        self.Layer1 = ResidualBlock(16,32,1)
        self.Layer2 = ResidualBlock(32,64,2)
        self.Layer3 = ResidualBlock(64,128,2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear()
        