import timm
import torch
from torch import nn
import torch.nn.functional as F

# class SakeNet(nn.Module):
#     def __init__(
#         self,
#     ):
#         super().__init__()
#         model_name = "convnext_base"
#         pretrained = True
#         base_model = timm.create_model(
#             model_name, num_classes=0, pretrained=pretrained, in_chans=3)
#         in_features = base_model.num_features
#         self.backbone = base_model
#         print("load imagenet model_name:", model_name)
#         print("load imagenet pretrained:", pretrained)

#         self.in_features = in_features
#         embedding_dim = 128
#         self.fc = nn.Linear(self.in_features, embedding_dim)

#     def get_embedding(self, image: torch.tensor) -> torch.tensor:
#         output = self.backbone(image)
#         output = self.fc(output)
#         return output


class SakeNet(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super().__init__()
        self.num_classes = num_classes

        model_name = "resnet18"
        pretrained = True
        base_model = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained, in_chans=3)
        in_features = base_model.num_features
        self.backbone = base_model
        print("load imagenet model_name:", model_name)
        print("load imagenet pretrained:", pretrained)

        self.in_features = in_features
        embedding_dim = 128
        self.embed = nn.Sequential(
            nn.Linear(self.in_features, embedding_dim),
        )
        self.cls = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.num_classes)
        )

    def get_embedding(self, image: torch.tensor) -> torch.tensor:
        x = self.backbone(image)
        x = self.embed(x)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.cls(x)
        return x


def model_factory(num_classes):
    model = SakeNet(num_classes)
    return model


if __name__ == "__main__":
    x = torch.rand(2, 3, 224, 224)
    model = SakeNet(2)
    model.eval()
    y = model.get_embedding(x)
    print(y.shape)
    y = model(x)
    print(y.shape)
