import os
import typing

import faiss
import numpy as np
import pandas as pd
import torch
from config import Config as C
from datasets import SakeDataset
from models import model_factory
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import setup_logger


def infer(data_loader: DataLoader, model: nn.Module) -> np.array:
    stream = tqdm(data_loader)
    model.eval()
    embedding = []
    for batch in stream:
        images = batch["image"].to(C.test_device, non_blocking=True).float()
        with torch.set_grad_enabled(mode=False):
            output = model.get_embedding(images)
            embedding.append(output.detach().cpu().numpy())
    embedding = np.concatenate(embedding)
    return embedding


class FaissKNeighbors:
    def __init__(self, index_name: str = "sake", k: int = 20) -> None:
        self.index = None
        self.d = None
        self.k = k
        self.index_name = str(index_name)

    def fit(self, X: np.array) -> None:
        X = X.copy(order="C")
        self.d = X.shape[1]
        # distance: cosine similarity
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(X.astype(np.float32))

    def save_index(self) -> None:
        faiss.write_index(self.index, self.index_name)
        print(f"{self.index_name} saved.")

    def read_index(self) -> None:
        self.index = faiss.read_index(self.index_name)
        self.d = self.index.d
        print(f"{self.index_name} read.")

    def predict(self, X: np.array) -> typing.Tuple:
        X = X.copy(order="C")
        X = np.reshape(X, (-1, self.d))
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        if X.shape[0] == 1:
            return distances[0], indices[0]
        else:
            return distances, indices


# 保存用ディレクトリ作成
os.makedirs(C.work_dir, exist_ok=True)
os.makedirs(C.features_dir, exist_ok=True)
os.makedirs(C.index_dir, exist_ok=True)

logger = setup_logger(str(C.work_dir.joinpath("test.log")))
logger.info("Test start.")

# アノテーション読み込み
df_cite = pd.read_csv(C.cite_filepath)
df_test = pd.read_csv(C.test_filepath)

cite_filenames = df_cite["cite_filename"].to_list()
df_cite["path"] = [str(C.cite_images_dir.joinpath(filename))
                   for filename in cite_filenames]
test_filenames = df_test["filename"].to_list()
df_test["path"] = [str(C.query_images_dir.joinpath(filename))
                   for filename in test_filenames]

# モデル読み込み
model = model_factory(C.num_classes)
model = model.to(C.test_device)
model_path = C.checkpoint_dir.joinpath(f"checkpoint_epoch_{C.test_epoch}.pth")
model.load_state_dict(torch.load(str(model_path))["model"])

# cite埋め込み作成
logger.info("Cite embedding start.")
cite_dataset = SakeDataset(
    image_filepaths=df_cite["path"].to_list()
)
cite_loader = DataLoader(
    cite_dataset,
    batch_size=C.test_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
cite_embedding = infer(cite_loader, model)
logger.info(f"Cite embedding shape: {cite_embedding.shape}")
np.save(C.cite_features_path, cite_embedding)
logger.info("Cite embedding end.")

# query埋め込み作成
logger.info("Query embedding start.")
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
query_embedding = infer(test_loader, model)
logger.info(f"Query embedding shape: {query_embedding.shape}")
np.save(C.query_features_path, query_embedding)
logger.info("Query embedding end.")


# インデックス作成
logger.info("Create index start.")
knn = FaissKNeighbors(C.index_path, k=20)
knn.fit(cite_embedding)
knn.save_index()
logger.info("Create index end.")

# クエリ実行
logger.info("Query start.")
idx2cite_gid = dict(zip(df_cite.index, df_cite["cite_gid"]))
cite_gids = []
for _query_embeding in tqdm(query_embedding):
    distance, pred = knn.predict(_query_embeding)
    _cite_gids = [str(idx2cite_gid[p]) for p in pred]
    cite_gids.append(" ".join(_cite_gids))
df_test["cite_gid"] = cite_gids
submission = df_test[["gid", "cite_gid"]]
submission.to_csv(C.work_dir.joinpath(
    f"submission_{C.exp}.csv"), index=False)
logger.info(f"submission shape: {submission.shape}")
df_test.to_csv(C.work_dir.joinpath(
    f"df_test_{C.exp}.csv"), index=False)
logger.info("Query end.")

logger.info("Test end.")
