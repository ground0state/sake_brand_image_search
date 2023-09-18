# %%
import torch

x = torch.load("../work_dirs/exp02/checkpoints/checkpoint_epoch_62.pth")
# %%
state_dict = x["model"]
# %%
