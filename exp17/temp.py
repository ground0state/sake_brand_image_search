# %%
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from models import model_factory
from matplotlib import pyplot as plt
from datasets import SakeDataset
from config import Config as C
import torch
import pandas as pd
import numpy as np
import faiss
import typing
import os
from sklearn import preprocessing

# %%


def infer(data_loader: DataLoader, model: nn.Module) -> np.array:
    stream = tqdm(data_loader)
    model.eval()
    embedding = []
    for batch in stream:
        images = batch["image"].to(C.test_device, non_blocking=True).float()
        with torch.set_grad_enabled(mode=False):
            output = F.softmax(model(images))
            embedding.append(output.detach().cpu().numpy())
    embedding = np.concatenate(embedding)
    return embedding
# %%


annotation_df = pd.read_csv(C.train_filepath)
train_filenames = annotation_df["filename"].to_list()
annotation_df["path"] = [str(C.query_images_dir.joinpath(filename))
                         for filename in train_filenames]
df_cite = pd.read_csv(C.cite_filepath)
df_test = pd.read_csv(C.test_filepath)

cite_filenames = df_cite["cite_filename"].to_list()
df_cite["path"] = [str(C.cite_images_dir.joinpath(filename))
                   for filename in cite_filenames]
test_filenames = df_test["filename"].to_list()
df_test["path"] = [str(C.query_images_dir.joinpath(filename))
                   for filename in test_filenames]
# %%
count_df = annotation_df.groupby("meigara").count()["gid"]
# %%
count_df.max()
# %%
test_dataset = SakeDataset(
    image_filepaths=df_test["path"].to_list()
)
test_loader = DataLoader(
    test_dataset,
    batch_size=C.test_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
model = model_factory(C.num_classes, C.embedding_dim)
model = model.to(C.test_device)
model_path = C.checkpoint_dir.joinpath(f"checkpoint_epoch_{C.test_epoch}.pth")
model.load_state_dict(torch.load(str(model_path))["model"])

# %%
query_embedding = infer(test_loader, model)

# %%
plt.hist(query_embedding.max(axis=1))
# %%

query_embedding
# %%
annotation_df
# %%
count_df.sort_values(ascending=False)
# %%
count_df[count_df < 30].shape
# %%
count_df.shape
# %%
count_df.mean()
# %%
le = preprocessing.LabelEncoder()
le.fit(annotation_df["meigara"])
# %%
le.classes_.shape
# %%
