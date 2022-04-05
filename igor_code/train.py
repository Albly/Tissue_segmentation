import os
import sys
import numpy as np
import pytorch_lightning as pl
import json
import shutil
from torchvision.models.segmentation import deeplabv3_resnet50

_folder_current = os.path.dirname(os.path.abspath(__file__))
# _folder = os.path.join(_folder_current, "..")
_folder = _folder_current
sys.path.append(_folder)
from hist_mask.model import DeepLab
from hist_mask.data import HistoDatasetRandom, get_dataloader, HistoDatasetAll
from hist_mask.pl import PLModel


def main(
        folder_model,
        folder_checkpoints,
        folder_logs,
        folder_data,
        max_epochs=100,
        val_ratio=0.1,
        random_seed=0,
        img_size=256,
        batch_size=32,
        ):

    pl.seed_everything(random_seed)
    np.random.seed(random_seed)

    names_all = [f.name.replace(".jpg", "") for f in os.scandir(folder_data) if 
        f.name.endswith(".jpg")]
    names = [n for n in names_all if not n.endswith("_mask") and n + "_mask" in names_all]
    
    indexes = np.arange(len(names))
    np.random.shuffle(indexes)
    # indexes = indexes[:len(indexes)//10]
    # indexes = indexes[:10]
    indexes_train = indexes[:int(np.ceil(len(indexes) * (1.-val_ratio)))]
    indexes_val = indexes[len(indexes_train):]

    dataset_train = HistoDatasetRandom(folder_data, [names[i] for i in indexes_train], 
        mode="train", augment=True, img_size=img_size, preload=True)
    dataset_val = HistoDatasetAll(folder_data, [names[i] for i in indexes_val], img_size=img_size)

    dataloader_train = get_dataloader(dataset_train, batch_size=batch_size, 
        shuffle=True, drop_last=True, num_workers=12)
    dataloader_val = get_dataloader(dataset_val, batch_size=batch_size, 
        shuffle=False, drop_last=False, num_workers=12)

    # model = DeepLab("resnet18", False, 1)
    model = DeepLab("mobilenet_v3_small", True, 1)
    # model = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)

    if os.path.isdir(folder_model):
        shutil.rmtree(folder_model)
    if os.path.isdir(folder_checkpoints):
        shutil.rmtree(folder_checkpoints)
    if os.path.isdir(folder_logs):
        shutil.rmtree(folder_logs)

    os.makedirs(folder_model, exist_ok=True)
    os.makedirs(folder_checkpoints, exist_ok=True)

    plmodel = PLModel(model, folder_model=folder_model, folder_checkpoints=folder_checkpoints,)

    logger = pl.loggers.TensorBoardLogger(folder_logs, name="", version="")
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=logger)
    trainer.fit(plmodel, dataloader_train, dataloader_val)

    return model


if __name__ == "__main__":
    model_name = "v3"
    main(
        f"./res/models/{model_name}",
        f"./res/checkpoints/{model_name}",
        f"./res/logs/{model_name}",
        "./res/data/train",
        img_size=512,
        batch_size=32,
    )