import torchvision
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from src.config import DEVICE

def get_model(num_classes=3):
    # SSD 모델 로드 (Pretrained)
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    # 모델의 클래스 수 수정
    model.head.classification_head.num_classes = num_classes

    return model.to(DEVICE)