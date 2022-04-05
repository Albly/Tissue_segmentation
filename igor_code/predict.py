import os
import sys
import numpy as np
import torch
# import matplotlib
# from PIL import Image
import cv2

_folder_current = os.path.dirname(os.path.abspath(__file__))
_folder = _folder_current
sys.path.append(_folder)
from hist_mask.model import DeepLab
from hist_mask.data import get_dataloader, HistoDatasetAll, recreate_full_image_list
# from hist_mask.pl import PLModel


def load_model(filename_checkpoint):
    model = DeepLab("mobilenet_v3_small", True, 1)
    model.load_state_dict(torch.load(filename_checkpoint))
    model.eval()
    return model


if __name__ == "__main__":

    folder_data = "./res/data/train"
    names_all = [f.name.replace(".jpg", "") for f in os.scandir(folder_data) if 
        f.name.endswith(".jpg")]
    names = [n for n in names_all if not n.endswith("_mask") and n + "_mask" in names_all]
    
    dataset = HistoDatasetAll(folder_data, names[:1], img_size=512)
    dataloader = get_dataloader(dataset, batch_size=32,
        shuffle=False, drop_last=False, num_workers=12)

    model = load_model("./res/checkpoints/v2/best.ckpt")

    predictions, indexes_all, indexes_2_all = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            images, _, indexes, indexes_2 = batch
            preds = model(images.to("gpu"))
            predictions.append(preds.cpu().numpy())
            indexes_all.append(indexes)
            indexes_2_all.append(indexes_2)
    predictions = np.concatenate(predictions)
    indexes = np.concatenate(indexes_all)
    indexes_2 = np.concatenate(indexes_2_all)

    predictions_full = recreate_full_image_list(predictions, indexes, indexes_2)

    cv2.imwrite("test.png", predictions_full)