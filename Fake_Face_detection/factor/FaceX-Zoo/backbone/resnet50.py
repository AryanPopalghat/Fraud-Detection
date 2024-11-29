import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Identity()  # Removing the final fully connected layer

    def forward(self, x):
        return self.model(x)
