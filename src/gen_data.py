import os
import numpy as np
import pandas as pd
from PIL import Image
import librosa
import joblib
import shutil
from tqdm import tqdm

def get_train_file_path(image_id):
    return "input/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "input/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def save_images(file_path):
    file_name = file_path.split('/')[-1].split('.npy')[0]
    waves = np.load(file_path).astype(np.float32) # (3, 4096)
    melspecs = []
    for j in range(3):
        melspec = librosa.feature.melspectrogram(waves[j] / max(waves[j]),
                                                 sr=4096, n_mels=128, fmin=20, fmax=2048)
        melspec = librosa.power_to_db(melspec)
        melspec = melspec.transpose((1, 0))
        melspecs.append(melspec)
    image = np.vstack(melspecs)
    np.save(OUT_DIR + file_name, image)


if __name__ == '__main__':
    print('generating images')

    OUT_DIR = "input/images/train/"

    train = pd.read_csv('input/training_labels.csv')
    # test = pd.read_csv('input/sample_submission.csv')

    train['file_path'] = train['id'].apply(get_train_file_path)
    # test['file_path'] = test['id'].apply(get_test_file_path)

    print(train.head())
    # display(test.head())

    _ = joblib.Parallel(n_jobs=10)(
        joblib.delayed(save_images)(file_path) for file_path in tqdm(train['file_path'].values)
    )

    shutil.make_archive(OUT_DIR, 'zip', OUT_DIR)
    shutil.rmtree(OUT_DIR)
