import os
import numpy as np
from re import L
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from .model import UNet, DeepLab
from .data import recreate_full_image_list
from . import loss


class ExtractStateDictCallback(pl.Callback):
    def __init__(self, folder_checkpoints, folder_model, filename, **kwargs):
        super(ExtractStateDictCallback, self).__init__(**kwargs)
        self.folder_checkpoints = folder_checkpoints
        self.folder_model = folder_model
        self.filename = filename
    def on_validation_epoch_end(self, *args, **kwargs):
        filename_checkpoint = os.path.join(self.folder_checkpoints, self.filename + ".ckpt")
        if os.path.isfile(filename_checkpoint):
            state_dict = {k[6:]: v for k, v in torch.load(filename_checkpoint)["state_dict"].items()}
            torch.save(state_dict, os.path.join(self.folder_model, self.filename + ".pth"))
        return


class PLModel(pl.LightningModule):
    def __init__(self, model, 
            optimizer_lr=1e-3,
            optimizer_weight_decay=0,
            scheduler_factor=0.1,
            scheduler_patience=20,
            scheduler_monitor="val_dice",
            folder_model=None,
            folder_checkpoints=None,
            ):
        super(PLModel, self).__init__()
        self.model = model
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_monitor = scheduler_monitor
        self.folder_model = folder_model
        self.folder_checkpoints = folder_checkpoints

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
            lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            ),
            "monitor": self.scheduler_monitor,
        }
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor=self.scheduler_monitor,
            mode="max",
            patience=10000,
        )
        os.makedirs(self.folder_checkpoints, exist_ok=True)
        os.makedirs(self.folder_model, exist_ok=True)
        checkpoint = ModelCheckpoint(
            dirpath=self.folder_checkpoints,
            monitor="val_dice",
            filename="best",
            mode="min",
            verbose=0,
        )
        checkpoint_last = ModelCheckpoint(
            dirpath=self.folder_checkpoints,
            filename="last",
            verbose=0,
            save_last=True,
        )
        model = ExtractStateDictCallback(self.folder_checkpoints, self.folder_model, "best")
        return [early_stop, checkpoint, checkpoint_last, model]

    def forward(self, x):
        return torch.sigmoid(self.model(x).squeeze(1))

    def training_step(self, train_batch, batch_idx):
        d = self.shared_step(train_batch)
        self.log("train_loss", d["loss"])
        return d

    def validation_step(self, val_batch, batch_idx):
        d = self.shared_step(val_batch)
        self.log("val_loss", d["loss"])
        return d

    def test_step(self, test_batch, batch_idx):
        d = self.shared_step(test_batch)
        self.log("test_loss", d["loss"])
        return d

    def shared_step(self, batch):
        if len(batch) == 4:
            images, masks, indexes, indexes_2 = batch
        else:
            images, masks = batch
            indexes, indexes_2 = None, None
        pred = self.forward(images)
        loss = F.binary_cross_entropy(pred, masks)
        d = {
            "loss": loss,
            # "pred": pred.detach().cpu().numpy(),
            # "mask": masks.cpu().numpy(),
            "pred": pred.detach().cpu(),
            "mask": masks.cpu(),
        }
        if indexes is not None:
            # d["index"] = indexes.cpu().numpy()
            # d["index_2"] = indexes_2.cpu().numpy()
            d["index"] = indexes.cpu().numpy()
            d["index_2"] = indexes_2.cpu().numpy()
        return d

    def get_val_scores(self, outputs):
        preds, masks, indexes, indexes_2 = [], [], [], []
        for out in outputs:
            preds.append(out["pred"])
            masks.append(out["mask"])
            if "index" in out:
                indexes.append(out["index"])
                indexes_2.append(out["index_2"])
        if len(indexes) > 0:
            indexes = np.concatenate(indexes)
            indexes_2 = np.concatenate(indexes_2)
            
        preds = torch.cat(preds)
        masks = torch.cat(masks)

        if len(indexes) > 0:
            preds_full = recreate_full_image_list(preds.numpy(), indexes, indexes_2)
            masks_full = recreate_full_image_list(masks.numpy(), indexes, indexes_2)
            scores = {}
            for p, m in zip(preds_full, masks_full):
                s = loss.calc_scores(torch.Tensor([p]), torch.Tensor([m]))
                for k, v in s.items():
                    scores.setdefault(k, []).append(v)
            scores = {k: np.mean(v) for k, v in scores.items()}
        else:
            scores = loss.calc_scores(preds, masks)
        return scores
            

    def log_val_scores(self, outputs, mode="val"):
        scores = self.get_val_scores(outputs)
        for k, v in scores.items():
            self.log(f"{mode}_scores/{k}", v, self.current_epoch)
        self.log_dict({f"{mode}_dice": scores["dice"]})

    def validation_epoch_end(self, outputs):
        self.log_val_scores(outputs, mode="val")
    def training_epoch_end(self, outputs):
        self.log_val_scores(outputs, mode="train")
    def test_epoch_end(self, outputs):
        self.log_val_scores(outputs, mode="test")

    def predict_step(self, batch):
        images, _, indexes, indexes_2 = batch
        pred = self.model(images)
        return {"pred": pred, "index": indexes, "index_2": indexes_2}