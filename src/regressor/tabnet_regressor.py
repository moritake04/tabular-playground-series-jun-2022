import torch.optim as optim
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor


class TNRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
        valid_X=None,
        valid_y=None,
    ):
        self.train_X = train_X.values
        self.train_y = train_y.values.reshape(-1, 1)

        if valid_X is not None:
            valid_X = valid_X.values
            valid_y = valid_y.values.reshape(-1, 1)
            self.eval_set = [(self.train_X, self.train_y), (valid_X, valid_y)]
        else:
            self.eval_set = [(self.train_X, self.train_y)]

        self.optimizer_fn = optim.__dict__[cfg["optimizer"]["name"]]
        self.optimizer_params = cfg["optimizer"]["params"]
        if cfg["scheduler"] is None:
            self.scheduler_fn = None
            self.scheduler_params = None
        else:
            self.scheduler_fn = optim.lr_scheduler.__dict__[cfg["scheduler"]["name"]]
            self.scheduler_params = cfg["scheduler"]["params"]

        self.cfg = cfg

    def train(self):
        if self.cfg["pretrain"]:
            unsupervised_model = TabNetPretrainer(
                optimizer_fn=self.optimizer_fn,
                optimizer_params=self.optimizer_params,
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                **self.cfg["tabnet_pretrain_params"],
            )
            unsupervised_model.fit(
                self.train_X,
                eval_set=[self.eval_set[-1][0]],
                **self.cfg["tabnet_pretrain_train_params"],
            )
        else:
            unsupervised_model = None
        self.tabnet = TabNetRegressor(
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            **self.cfg["tabnet_params"],
        )
        self.tabnet.fit(
            self.train_X,
            self.train_y,
            eval_set=self.eval_set,
            eval_name=["train", "valid"],
            from_unsupervised=unsupervised_model,
            **self.cfg["tabnet_train_params"],
        )

    def predict(self, test_X):
        preds = self.tabnet.predict(test_X.values)
        return preds
