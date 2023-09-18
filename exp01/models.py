import timm
import torch
from torch import nn


class SakeNet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        model_name = "convnext_base"
        pretrained = True
        base_model = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained, in_chans=3)
        in_features = base_model.num_features
        self.backbone = base_model
        print("load imagenet model_name:", model_name)
        print("load imagenet pretrained:", pretrained)

        self.in_features = in_features
        embedding_dim = 128
        self.fc = nn.Linear(self.in_features, embedding_dim)

    def get_embedding(self, image: torch.tensor) -> torch.tensor:
        output = self.backbone(image)
        output = self.fc(output)
        return output
