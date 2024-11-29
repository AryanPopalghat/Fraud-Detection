from torchvision.models.detection import RetinaNet
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn

def get_model(name, max_size, device):
    if name == "resnet50_2020-07-20":
        model = retinanet_resnet50_fpn(pretrained=True)
    else:
        raise ValueError("Model name not recognized.")
    model.to(device)
    return model
