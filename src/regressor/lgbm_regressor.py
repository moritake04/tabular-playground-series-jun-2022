import warnings

import lightgbm as lgb

warnings.simplefilter("ignore", UserWarning)


class LGBMRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
        valid_X=None,
        valid_y=None,
        categorical_feature=None,
    ):
        # create datasets
        self.lgbm_train = lgb.Dataset(
            train_X, train_y, categorical_feature=categorical_feature
        )
        if valid_X is not None:
            self.lgbm_valid = lgb.Dataset(
                valid_X, valid_y, categorical_feature=categorical_feature
            )
            self.valid_sets = [self.lgbm_train, self.lgbm_valid]
        else:
            self.lgbm_valid = None
            self.valid_sets = [self.lgbm_train]

        self.categorical_feature = categorical_feature
        self.cfg = cfg

    def train(self):
        self.lgbm = lgb.train(
            params=self.cfg["lgbm_params"],
            train_set=self.lgbm_train,
            valid_sets=self.valid_sets,
            categorical_feature=self.categorical_feature,
            **self.cfg["lgbm_train_params"]
        )

    def predict(self, test_X):
        preds = self.lgbm.predict(test_X, num_iterations=self.lgbm.best_iteration)
        return preds
