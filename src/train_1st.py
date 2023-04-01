import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from pytorch_lightning import seed_everything
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from regressor import LGBMRegressor, NNRegressor, OneDCNNRegressor, TNRegressor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    args = parser.parse_args()
    return args


def train_and_predict(cfg, train_X, train_y, test_X, valid_X=None, valid_y=None):
    if cfg["model"] == "lgbm":
        model = LGBMRegressor(cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y)
    elif cfg["model"] == "nn":
        model = NNRegressor(cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y)
    elif cfg["model"] == "tabnet":
        model = TNRegressor(cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y)
    elif cfg["model"] == "onedcnn":
        model = OneDCNNRegressor(
            cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y
        )

    model.train()
    test_preds = model.predict(test_X)

    if valid_X is None:
        return test_preds
    else:
        valid_preds = model.predict(valid_X)
        return test_preds, valid_preds


def one_fold(fold_n, skf, cfg, train_X, train_y, test_X):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    train_indices, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    train_X_cv, train_y_cv = (
        train_X.iloc[train_indices],
        train_y[train_indices],
    )
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices],
        train_y[valid_indices],
    )

    # train and valid
    test_preds, valid_preds = train_and_predict(
        cfg,
        train_X_cv,
        train_y_cv,
        test_X,
        valid_X=valid_X_cv,
        valid_y=valid_y_cv,
    )

    rmse = np.sqrt(mean_squared_error(valid_y_cv, valid_preds))

    torch.cuda.empty_cache()

    print(f"[fold_{fold_n}] finished, rmse:{rmse}")

    return valid_preds, test_preds, rmse


def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    data = pd.read_csv("../data/input/data_nan_flg.csv")
    sample_submission = pd.read_csv("../data/output/sample_submission.csv")

    # imputation of missing-values
    # F4_ features
    f4_features = []
    imputed_data_f4 = data.copy()
    valid_rmse_list_all = []  # used only cross validation
    for feature in data.columns:
        if feature[:3] == "F_4":
            f4_features.append(feature)
    f4_features.append("number_of_null_f4")
    for feature in tqdm(f4_features):
        tqdm.write(feature)
        # null -> test, not null -> train
        data_f4 = data.loc[:, ["row_id"] + f4_features]
        test = data_f4.loc[data_f4[feature].isnull()]
        if len(test) == 0:
            continue
        train = data_f4.loc[(data_f4[feature].notnull())]

        # train / test split
        train_y = train[feature].reset_index(drop=True)
        train_X = train.drop(["row_id", feature, feature + "_nan"], axis=1).reset_index(
            drop=True
        )
        test_X = test.drop(["row_id", feature, feature + "_nan"], axis=1)

        # fillna
        if cfg["model"] != "lgbm":
            for i in train_X.columns:
                train_X[i] = train_X[i].fillna(data[i].mean())
                test_X[i] = test_X[i].fillna(data[i].mean())

        # "number_of_null_f4" for explanatory variables only.
        test_X["number_of_null_f4"] -= 1

        if cfg["general"]["fold"] is None:
            # train all data
            preds = train_and_predict(cfg, train_X, train_y, test_X)
            imputed_data_f4.loc[test.index, feature] = preds
        else:
            # cross validation
            skf = KFold(
                n_splits=cfg["general"]["n_splits"],
                shuffle=True,
                random_state=cfg["general"]["seed"],
            )
            test_preds_list = []
            valid_rmse_list = []
            results = Parallel(n_jobs=cfg["general"]["n_jobs"])(
                delayed(one_fold)(fold_n, skf, cfg, train_X, train_y, test_X)
                for fold_n in cfg["general"]["fold"]
            )
            for _, test_preds, rmse in results:
                test_preds_list.append(test_preds)
                valid_rmse_list.append(rmse)
            print(valid_rmse_list)
            imputed_data_f4.loc[test.index, feature] = np.mean(test_preds_list, axis=0)
            valid_rmse_list_all.append(np.mean(valid_rmse_list))

    # show rmse in each future
    for i, feature in enumerate(f4_features):
        if feature[:3] == "F_4" and feature[-3:] != "nan":
            print(f"{feature} rmse: {valid_rmse_list_all[i]}")
        else:
            continue

    # other features
    other_features = list(set(data.columns) - set(f4_features))
    # imputation of missing-values (mean)
    for i in other_features:
        data[i] = data[i].fillna(data[i].mean())

    # f4_features values to data
    for i in f4_features:
        data[i] = imputed_data_f4[i]

    for i in tqdm(range(len(sample_submission))):
        row_id, row_col = sample_submission["row-col"].iloc[i].split("-")
        imputed_value = data.loc[int(row_id), row_col]
        sample_submission.loc[i, "value"] = imputed_value
    sample_submission.to_csv(cfg["general"]["save_name"], index=False)


if __name__ == "__main__":
    main()
