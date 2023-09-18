import os
from pathlib import Path

# 実行中のファイルの絶対パスを取得
current_file_path = os.path.abspath(__file__)
# 絶対パスからディレクトリ名を取得
current_directory = os.path.basename(os.path.dirname(current_file_path))


class Config:
    # exp paramters
    exp = current_directory
    work_dirs = Path("../work_dirs")
    seed = 0
    # load path
    data_dir = Path("/media/data/sake_brand_image_search")
    annotation_dir = data_dir.joinpath("data")
    cite_images_dir = data_dir.joinpath("cite_images")
    query_images_dir = data_dir.joinpath("query_images")
    cite_filepath = annotation_dir.joinpath("cite.csv")
    train_filepath = annotation_dir.joinpath("train.csv")
    test_filepath = annotation_dir.joinpath("test.csv")
    sub_filepath = annotation_dir.joinpath("sample_submission.csv")
    # save path
    work_dir = Path(f"../work_dirs/{exp}")
    features_dir = work_dir.joinpath("features")
    cite_features_path = features_dir.joinpath("cite_embedding.npy")
    query_features_path = features_dir.joinpath("query_embedding.npy")
    index_dir = work_dir.joinpath("index")
    index_path = index_dir.joinpath("sake.index")
    log_dir = work_dir.joinpath("tf_logs")
    checkpoint_dir = work_dir.joinpath("checkpoints")
    # train 設定
    num_classes = 2499
    val_group = 2
    num_epochs = 10
    train_batch_size = 32
    val_batch_size = 128
    learning_rate = 1e-4
    eta_min = 0  # learning_rate/100.0
    log_interval = 50
    eval_interval = 10
    num_last_epochs = 10
    embedding_dim = 1280  # 768
    train_device = "cuda"
    # test 設定
    test_batch_size = 128
    test_epoch = 10
    test_device = "cuda"


if __name__ == "__main__":
    # クラス変数の一覧を取得
    class_variables = {var: getattr(Config, var) for var in dir(Config)
                       if not callable(getattr(Config, var))
                       and not var.startswith("__")}
    print(class_variables)
