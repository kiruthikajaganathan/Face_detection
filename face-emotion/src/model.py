import torch.nn as nn
import torchvision.models as models

LABELS = ['angry','disgust','fear','happy','sad','surprise','neutral']

def get_model(num_classes=len(LABELS), pretrained=True, dropout=0.2):
    model = models.mobilenet_v2(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model
