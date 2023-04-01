import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from joblib.externals.loky.backend.context import get_context
from pytorch_lightning import Trainer

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred_y, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred_y, y))
        return loss


class simpleNN(pl.LightningModule):
    def __init__(self, input_size, cfg):
        super().__init__()

        self.cfg = cfg

        """
        f4_features_mean = [
            0.32660781577435904,
            -0.3308645443678977,
            -0.08579204399264272,
            -0.1954881776610604,
            0.33305974884806966,
            0.3359677798524754,
            0.0037725123029517037,
            0.33443312565657857,
            -0.07184194934428197,
            -0.0798537370491984,
            0.038281550696440764,
            0.5518997711372439,
            0.3335091433868465,
            0.3300473562558188,
            0.037223107205320724,
        ]
        self.f4_features_mean = []
        for feature in features:
            self.f4_features_mean.append(f4_features_mean[int(feature.split("_")[-1])])
        """

        if cfg["criterion"] == "RMSELoss":
            self.criterion = RMSELoss()
        else:
            self.criterion = nn.__dict__[cfg["criterion"]]()

        self.regressor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch

        """
        # randomly missing
        for _ in range(int(self.cfg["train_loader"]["batch_size"] * 0.05)):
            target_index = torch.randint(
                0, self.cfg["train_loader"]["batch_size"], (1,)
            )
            target_column = torch.randint(0, 13, (1,))
            nan_flg_column = target_column + 14
            X[target_index, target_column] = self.f4_features_mean[target_column]
            X[target_index, nan_flg_column] = 1
        """

        pred_y = self.forward(X).squeeze()
        loss = self.criterion(pred_y, y)
        return loss

    def training_epoch_end(self, outputs):
        loss_list = [x["loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X).squeeze()
        loss = self.criterion(pred_y, y)
        return {"valid_loss": loss}

    def validation_epoch_end(self, outputs):
        loss_list = [x["valid_loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("valid_avg_loss", avg_loss, prog_bar=True)
        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        pred_y = self.forward(X)
        return pred_y

    def configure_optimizers(self):
        optimizer = optim.__dict__[self.cfg["optimizer"]["name"]](
            self.parameters(), **self.cfg["optimizer"]["params"]
        )
        if self.cfg["scheduler"] is None:
            return [optimizer]
        else:
            scheduler = optim.lr_scheduler.__dict__[self.cfg["scheduler"]["name"]](
                optimizer, **self.cfg["scheduler"]["params"]
            )
            return [optimizer], [scheduler]


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        if y is None:
            self.X = X
            self.y = torch.zeros(len(self.X), dtype=torch.float32)
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y


"""
class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        len_dataset,
        labels,
        num_samples=None,
    ):
        self.labels = [3 if label >= 3 else label for label in labels]
        self.indices = list(range(len_dataset))
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self.labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
"""


class NNRegressor:
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        input_size = len(train_X.columns)
        # create datasets
        train_X = torch.tensor(train_X.values, dtype=torch.float32)
        train_y = torch.tensor(train_y.values, dtype=torch.float32)
        self.train_dataset = TableDataset(train_X, train_y)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            multiprocessing_context=get_context("loky"),
            **cfg["train_loader"],
        )
        if valid_X is None:
            self.valid_dataloader = None
        else:
            valid_X = torch.tensor(valid_X.values, dtype=torch.float32)
            valid_y = torch.tensor(valid_y.values, dtype=torch.float32)
            self.valid_dataset = TableDataset(valid_X, valid_y)
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                multiprocessing_context=get_context("loky"),
                **cfg["valid_loader"],
            )

        self.callbacks = []

        if cfg["early_stopping"] is not None:
            self.callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss",
                    patience=cfg["early_stopping"]["patience"],
                )
            )

        self.model = simpleNN(input_size, cfg)
        self.cfg = cfg

    def train(self):
        self.trainer = Trainer(callbacks=self.callbacks, **self.cfg["pl_params"])
        if self.valid_dataloader is None:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
            )
        else:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.valid_dataloader,
            )

    def predict(self, test_X):
        preds = []
        test_X = torch.tensor(test_X.values, dtype=torch.float32)
        test_dataset = TableDataset(test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            multiprocessing_context=get_context("loky"),
            **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(self.model, dataloaders=test_dataloader)
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds
