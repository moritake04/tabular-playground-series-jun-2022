import pandas as pd
from tqdm import tqdm


def main():
    # read csv
    data = pd.read_csv("../data/input/data.csv")

    # create nan flg
    for feature in data.columns:
        data[f"{feature}_nan"] = data[feature].isna() * 1

    # retrieve f1, f3, f4 features
    f1_features = []
    for feature in data.columns:
        if feature[:3] == "F_1" and feature[-3:] != "nan":
            f1_features.append(feature)
    data_f1 = data[f1_features]

    f3_features = []
    for feature in data.columns:
        if feature[:3] == "F_3" and feature[-3:] != "nan":
            f3_features.append(feature)
    data_f3 = data[f3_features]

    f4_features = []
    for feature in data.columns:
        if feature[:3] == "F_4" and feature[-3:] != "nan":
            f4_features.append(feature)
    data_f4 = data[f4_features]

    # create number_of_null (all)
    data["number_of_null_all"] = -1
    for i in tqdm(range(len(data))):
        data.loc[i, "number_of_null_all"] = data.iloc[i].isnull().sum()

    # create number_of_null (f1)
    data["number_of_null_f1"] = -1
    for i in tqdm(range(len(data))):
        data.loc[i, "number_of_null_f1"] = data_f1.iloc[i].isnull().sum()

    # create number_of_null (f3)
    data["number_of_null_f3"] = -1
    for i in tqdm(range(len(data))):
        data.loc[i, "number_of_null_f3"] = data_f3.iloc[i].isnull().sum()

    # create number_of_null (f4)
    data["number_of_null_f4"] = -1
    for i in tqdm(range(len(data))):
        data.loc[i, "number_of_null_f4"] = data_f4.iloc[i].isnull().sum()

    # save
    data.to_csv("../data/input/data_nan_flg.csv", index=False)


if __name__ == "__main__":
    main()
