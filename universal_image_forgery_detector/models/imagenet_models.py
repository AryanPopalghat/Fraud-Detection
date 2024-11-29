from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg19
from torchvision import transforms
from PIL import Image
import torch 
import torch.nn as nn 

model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vgg11': vgg11,
    'vgg19': vgg19
}

CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50" : 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "vgg11" : 4096,
    "vgg19" : 4096
}

class ImagenetModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ImagenetModel, self).__init__()

        self.model = model_dict[name](pretrained=True)
        self.fc = nn.Linear(CHANNELS[name], num_classes) # manually define a fc layer here

    def forward(self, x):
        feature = self.model(x)["penultimate"]
        return self.fc(feature)
