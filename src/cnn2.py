from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.nn as nn

num_classes = 1000

class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.conv1,
        self.bn1,
        self.relu,
        self.maxpool,

        self.layer1,
        self.layer2,
            
        self.layer3,
        self.layer4,
        self.avgpool

        #self.fc.to('cuda')

    def forward(self, x):
        #x = self.seq1.to('cuda')
        return self.fc(x.view(x.size(0), -1))

class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')
        #.to('cuda:1')

        self.fc.to('cuda:1')
        #.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        #.to('cuda:1')
        return self.fc(x.view(x.size(0), -1))
