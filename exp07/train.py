import os
from os.path import join

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config as C
from datasets import SakeDataset
from losses import CenterLoss
from models import model_factory
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter  # TensorBoardのためのライブラリ
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from utils import save_checkpoint, set_seed, setup_logger

# シードを設定
set_seed(C.seed)

# アノテーション読み込み
annotation_df = pd.read_csv(C.train_filepath)
train_filenames = annotation_df["filename"].to_list()
annotation_df["path"] = [str(C.query_images_dir.joinpath(filename))
                         for filename in train_filenames]
le = preprocessing.LabelEncoder()
le.fit(annotation_df["meigara"])
annotation_df["target"] = le.transform(annotation_df["meigara"])

# データの分割
kf = KFold(n_splits=10, shuffle=True, random_state=C.seed)
split_n = C.val_group  # 実験対象のfold
train_index, val_index = list(kf.split(range(len(annotation_df))))[split_n]
train_df = annotation_df.iloc[train_index]
val_df = annotation_df.iloc[val_index]

# 保存用ディレクトリ作成
os.makedirs(C.work_dir, exist_ok=True)
os.makedirs(C.checkpoint_dir, exist_ok=True)
os.makedirs(C.log_dir, exist_ok=False)  # 実験ミスを防ぐためにexist_ok=False

logger = setup_logger(join(C.work_dir, "train.log"))

train_dataset = SakeDataset(
    image_filepaths=train_df["path"].to_list(),
    labels=train_df["target"].to_list(),
    is_train=True)
val_dataset = SakeDataset(
    image_filepaths=val_df["path"].to_list(),
    labels=val_df["target"].to_list(),
    is_train=False)


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch]
                         ) if 'label' in batch[0] else None
    return images, labels


train_loader = DataLoader(
    train_dataset,
    batch_size=C.train_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=2)
val_loader = DataLoader(
    val_dataset,
    batch_size=C.val_batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2)

# モデル、損失関数、最適化関数の設定
model = model_factory(C.num_classes, C.embedding_dim)
model = model.to(C.train_device)

criterion = nn.CrossEntropyLoss(reduction="mean")

optimizer = optim.AdamW(model.parameters(), lr=C.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=C.num_epochs*len(train_loader), eta_min=C.eta_min)
# scheduler = get_cosine_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=500, num_training_steps=C.num_epochs*len(train_loader))

# 訓練ループ
logger.info("Training start.")
# TensorBoardのためのライターをセットアップ
writer = SummaryWriter(log_dir=C.log_dir)

num_steps = len(train_loader)
for epoch in range(1, C.num_epochs+1):
    # Train loop
    model.train()
    total_num_data = 0
    total_loss = 0.0
    for step, (inputs, labels) in enumerate(train_loader, start=1):
        inputs, labels = inputs.to(C.train_device), labels.to(C.train_device)
        outputs, embed = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_num_data += inputs.size(0)

        if step % C.log_interval == 0:
            running_loss = total_loss / total_num_data
            logger.info(
                f"Epoch [{epoch}/{C.num_epochs}], Step [{step}/{num_steps}], Running Loss: {running_loss:.4f}")
            # TensorBoardに訓練損失を記録
            writer.add_scalar("Train/Loss", running_loss,
                              (epoch-1) * num_steps + step)
            lr_ = scheduler.get_last_lr()[0]
            writer.add_scalar("Train/Lr", lr_,
                              (epoch-1) * num_steps + step)
            total_num_data = 0
            total_loss = 0.0

        # 学習率スケジューラーのステップ
        scheduler.step()

    # if (epoch > (C.num_epochs - C.num_last_epochs)) or (epoch % C.eval_interval == 0):
    if True:  # Hack: temporalなコード
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(
                    C.train_device), labels.to(C.train_device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / len(val_loader.dataset)

        logger.info(
            f"Epoch [{epoch}/{C.num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        # TensorBoardに検証損失を記録
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        # TensorBoardにエポックごとの検証精度を記録
        writer.add_scalar("Val/Accuracy", val_accuracy, epoch)

    if epoch > (C.num_epochs - C.num_last_epochs):
        # チェックポイントの保存
        checkpoint_path = join(
            C.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(epoch, model, optimizer,
                        scheduler, path=checkpoint_path)

logger.info("Training complete.")
