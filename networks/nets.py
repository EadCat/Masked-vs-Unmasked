import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, num_classes=1):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(1000, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 24)
        self.layer5 = nn.Linear(24, num_classes)
        self.final = nn.Softmax(dim=1)

    def forward(self, x):
        output = F.relu(self.layer1(x))
        output = F.relu(self.layer2(output))
        output = F.relu(self.layer3(output))
        output = F.relu(self.layer4(output))
        output = self.layer5(output)
        # output = self.final(self.layer5(output))
        return output


# class Classifier(nn.Module):
#     def __init__(self, num_classes=1):
#         super(Classifier, self).__init__()
#         self.layer1 = nn.Conv2d(1000, 512, kernel_size=1, stride=1)
#         self.layer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
#         self.layer3 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
#         self.layer4 = nn.Conv2d(128, 24, kernel_size=1, stride=1)
#         self.layer5 = nn.Conv2d(24, num_classes, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         output = self.layer1(x)
#         output = self.layer2(output)
#         output = self.layer3(output)
#         output = self.layer4(output)
#         output = self.layer5(output)
#         return output


class ResNet34(nn.Module):
    def __init__(self, netend, pretrain=False):
        super(ResNet34, self).__init__()
        self.backbone = models.resnet34(pretrained=pretrain)
        self.classifier = netend

    def forward(self, x):
        output = self.backbone(x)
        output = self.classifier(output)
        return output


class ResNet50(nn.Module):
    def __init__(self, netend, pretrain=False):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrain)
        self.classifier = netend

    def forward(self, x):
        output = self.backbone(x)
        output = self.classifier(output)
        return output










