import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=False)
state_dict = torch.load('E:/Aiwi/Project/factor/FaceX-Zoo/backbone/resnet50-19c8e357.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()
