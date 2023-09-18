# %%
import timm
import torch
from torch import nn
import torch.nn.functional as F


class SakeNet(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_dim
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        model_name = "tf_efficientnet_b0_ns"
        # model_name = "swin_small_patch4_window7_224.ms_in22k"
        pretrained = True
        base_model = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained, in_chans=3)
        in_features = base_model.num_features
        self.backbone = base_model
        print("load imagenet model_name:", model_name)
        print("load imagenet pretrained:", pretrained)

        self.in_features = in_features
        # self.embed = nn.Sequential(
        #     nn.Linear(self.in_features, embedding_dim),
        # )
        self.cls = nn.Sequential(
            # nn.BatchNorm1d(embedding_dim),
            # nn.ReLU(),
            nn.Linear(self.in_features, self.num_classes)
        )

    def get_embedding(self, image: torch.tensor) -> torch.tensor:
        x = self.backbone(image)
        # x = self.embed(x)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        y = self.cls(x)
        if self.training:
            return y, x
        else:
            return y


def model_factory(num_classes, embedding_dim):
    model = SakeNet(num_classes, embedding_dim)
    return model


# if __name__ == "__main__":
#     x = torch.rand(2, 3, 224, 224)
#     model = SakeNet(1000, 768)
#     model.eval()
#     y = model.get_embedding(x)
#     print(y.shape)
#     y = model(x)
#     print(y.shape)
#     print("aaa")

# %%
# model_name = "mobilenetv3_large_100"
# model_name = "inception_v4"
# model_name = "swin_small_patch4_window7_224.ms_in22k"
model_name = "tf_efficientnet_b0_ns"
pretrained = False
base_model = timm.create_model(
    model_name, num_classes=0, pretrained=pretrained, in_chans=3)
base_model.num_features
# %%
