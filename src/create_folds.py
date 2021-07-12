# mostly taken from https://www.kaggle.com/yasufuminakama/g2net-spectrogram-generation-train
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


def get_train_file_path(image_id):
    return "input/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id
    )


if __name__ == "__main__":
    print("creating folds")

    seed = 42
    n_fold = 5

    train = pd.read_csv("input/training_labels.csv")
    train["file_path"] = train["id"].apply(get_train_file_path)

    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for n, (train_index, val_index) in enumerate(skf.split(train, train["target"])):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    print(train.groupby(["fold", "target"]).size())

    train.to_csv("input/train_folds.csv")

    print(train.head())
