# %%
from PIL import Image
from matplotlib import pyplot as plt
import os
import typing

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config as C
# %%


class FaissKNeighbors:
    def __init__(self, index_name: str = "sake", k: int = 20) -> None:
        self.index = None
        self.d = None
        self.k = k
        self.index_name = str(index_name)

    def fit(self, X: np.array) -> None:
        X = X.copy(order="C")
        X = X.astype(np.float32)
        # faiss.normalize_L2(X)
        self.d = X.shape[1]
        # distance: cosine similarity
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(X)

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
        X = X.astype(np.float32)
        # faiss.normalize_L2(X)
        distances, indices = self.index.search(X, k=self.k)
        if X.shape[0] == 1:
            return distances[0], indices[0]
        else:
            return distances, indices


class Logger:
    def info(self, s):
        print(s)


dir_list = [
    # "exp13_0",
    # "exp13_1",
    # "exp13_2",
    # "exp13_3",
    # "exp13_4",
    "exp15_0",
    "exp15_1",
    "exp15_2",
    "exp15_3",
    "exp15_4",
    "exp18_0",
    "exp18_1",
    "exp18_2",
    "exp18_3",
    "exp18_4",
]


query_embedding = []
cite_embedding = []
for dir_name in dir_list:
    query_embedding_ = np.load(
        f"../work_dirs/{dir_name}/features/query_embedding.npy")
    faiss.normalize_L2(query_embedding_)
    query_embedding.append(query_embedding_)

    cite_embedding_ = np.load(
        f"../work_dirs/{dir_name}/features/cite_embedding.npy")
    faiss.normalize_L2(cite_embedding_)
    cite_embedding.append(cite_embedding_)

query_embedding = np.concatenate(query_embedding, axis=1)
cite_embedding = np.concatenate(cite_embedding, axis=1)

# %%
logger = Logger()
# インデックス作成
logger.info("Create index start.")
knn = FaissKNeighbors("./merged.index", k=20)
knn.fit(cite_embedding)
knn.save_index()
logger.info("Create index end.")


# アノテーション読み込み
df_cite = pd.read_csv(C.cite_filepath)
df_test = pd.read_csv(C.test_filepath)

cite_filenames = df_cite["cite_filename"].to_list()
df_cite["path"] = [str(C.cite_images_dir.joinpath(filename))
                   for filename in cite_filenames]
test_filenames = df_test["filename"].to_list()
df_test["path"] = [str(C.query_images_dir.joinpath(filename))
                   for filename in test_filenames]

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


# %%
submission.to_csv(f"merged_submission.csv", index=False)
logger.info(f"submission shape: {submission.shape}")
df_test.to_csv(f"merged_df_test.csv", index=False)
logger.info("Query end.")

logger.info("Test end.")

# %%
