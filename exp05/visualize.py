# %%
import pandas as pd
from config import Config as C
from matplotlib import pyplot as plt
from PIL import Image


def view_result_bygid(df_test: pd.DataFrame, gid: int) -> None:
    pred_gids = df_test.loc[df_test["gid"] == gid, "cite_gid"].values[0]
    pred_gids = pred_gids.split()

    query_path = df_test.loc[df_test["gid"] == gid, "path"].values[0]
    paths = []
    paths.append(query_path)
    cite_paths = [str(C.cite_images_dir.joinpath(path + ".jpg"))
                  for path in pred_gids]
    paths.extend(cite_paths)

    figs, axs = plt.subplots(nrows=7, ncols=3, figsize=(10, 20))
    for i, path in enumerate(paths):
        img = Image.open(path)
        i_row = int(i / 3)
        i_col = i % 3
        axs[i_row, i_col].imshow(img)
        gid = path.split("/")[-1].replace(".jpg", "")
        if i == 0:
            title = f"query data gid:{gid}"
            color = "red"
        else:
            title = f"rank: {i}, cite_gid:{gid}"
            color = "black"

        axs[i_row, i_col].set_title(title, color=color)
        axs[i_row, i_col].grid(False)
        axs[i_row, i_col].axis("off")
    plt.show()


# %%
df_test = pd.read_csv(C.work_dir.joinpath(f"df_test_{C.exp}.csv"))
# %%
view_result_bygid(df_test, gid=200108161)

# %%
