import pandas as pd
import numpy as np


def compute_mrr(rank_list):
    """
    Mean Reciprocal Rank (MRR)の計算

    Args:
        rank_list (list): リストのリスト。各内部リストには、ユーザーの推薦アイテムのランクが含まれています。
        ランクは、推薦アイテムのリスト内の位置によって決定されます。ランク外はinfとして扱われRRは0となります。

    Returns:
        float: The computed MRR value.
    """
    reciprocal_ranks = [1.0/rank if rank !=
                        float('inf') else 0.0 for rank in rank_list]
    return np.mean(reciprocal_ranks)


def compute_rank_list(recommendations, true_items, topk=20):
    """
    真の値のリストと推薦値のリストからランクリストを計算します。

    Args:
        recommendations (list): 推薦アイテムのリストのリスト。各内部リストには、ユーザーの推薦アイテムが含まれています。
        true_items (list): 真のアイテムのリストのリスト。各内部リストには、ユーザーの真のアイテムが含まれています。
        topk (int): ランク計算に用いる上位k件
    Returns:
        list: 最初の真のアイテムが見つかったときの推薦アイテムのランクのリスト。topk件に満たない場合はinfを追加します。
    """
    rank_list = []
    for rec_item_list, true_item_list in zip(recommendations, true_items):
        found = False
        # topk件推薦アイテムを計算対象とする
        rec_item_list = rec_item_list.split()[:topk]
        true_item_list = true_item_list.split()
        # 推薦アイテムを1位から見ていき、最初に真のアイテムが見つかったときのランクを記録する
        for idx, rec in enumerate(rec_item_list):
            if rec in true_item_list:
                # +1 because ranks start from 1, not 0
                rank_list.append(idx + 1)
                found = True
                break
        if not found:
            # if no true item found in recommendations
            rank_list.append(float('inf'))
    return rank_list


df_sub = pd.read_csv("sample_submission.csv")  # 提出ファイル
df_true = pd.read_csv("true.csv")  # 正解データのファイル
rank_list = compute_rank_list(
    df_sub["cite_gid"], df_true["cite_gid"].to_list())
score = compute_mrr(rank_list)
