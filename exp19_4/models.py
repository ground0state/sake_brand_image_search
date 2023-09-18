# %%
import timm
import torch
from torch import nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps).view(x.size(0), -1)

    def gem(self, x, p=3, eps=1e-6):
        # x = x.permute(0, 3, 1, 2)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class SakeCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_dim,
        pretrained=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # model_name = "tf_efficientnet_b0_ns"
        model_name = "convnext_small_in22k"
        pretrained = pretrained
        base_model = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained, in_chans=3)
        in_features = base_model.num_features
        self.backbone = base_model
        self.backbone.global_pool = GeM(p=4)
        print("load imagenet model_name:", model_name)
        print("load imagenet pretrained:", pretrained)

        self.in_features = in_features
        self.cls = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.in_features, self.num_classes),
        )

    def get_embedding(self, image: torch.tensor) -> torch.tensor:
        x = self.backbone(image)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        y = self.cls(x)
        if self.training:
            return y, x
        else:
            return y


class SakeSwin(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_dim,
        pretrained=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        model_name = "swin_small_patch4_window7_224.ms_in22k"
        pretrained = pretrained
        base_model = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained, in_chans=3)
        in_features = base_model.num_features
        self.backbone = base_model
        print("load imagenet model_name:", model_name)
        print("load imagenet pretrained:", pretrained)

        self.in_features = in_features
        self.cls = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.in_features, self.num_classes)
        )

    def get_embedding(self, image: torch.tensor) -> torch.tensor:
        x = self.backbone(image)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        y = self.cls(x)
        if self.training:
            return y, x
        else:
            return y


def model_factory(num_classes, embedding_dim, pretrained=True):
    model = SakeSwin(num_classes, embedding_dim, pretrained)
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
# model_name = "tf_efficientnet_b0_ns"
# model_name = "convnext_small_in22ft1k"
# pretrained = False
# base_model = timm.create_model(
#     model_name, num_classes=0, pretrained=pretrained, in_chans=3)
# base_model.num_features

# %%
