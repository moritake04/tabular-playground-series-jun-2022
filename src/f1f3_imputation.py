import pandas as pd
from tqdm import tqdm


def main():
    # read csv
    data = pd.read_csv("../data/input/data.csv")
    submission = pd.read_csv("../data/output/submission2_seed42-_averaging.csv")

    # compute f4 features
    f4_features = []
    for feature in data.columns:
        if feature[:3] == "F_4":
            f4_features.append(feature)

    """
    # extract features with skewed distributions.
    skewed_features = ["F_1_7", "F_1_12", "F_1_13", "F_3_19", "F_3_21"]

    # skewed features -> median
    for i in skewed_features:
        data[i] = data[i].fillna(data[i].median())
    """

    # other features -> mean
    other_features = list(set(data.columns) - set(f4_features))
    print(other_features)
    for i in other_features:
        data[i] = data[i].fillna(data[i].mean())

    # create submission
    for i in tqdm(range(len(submission))):
        row_id, row_col = submission["row-col"].iloc[i].split("-")
        if row_col[:3] == "F_4":
            continue
        imputed_value = data.loc[int(row_id), row_col]
        submission.loc[i, "value"] = imputed_value
    submission.to_csv("../data/output/submission2_seed42-_averaging_.csv", index=False)


if __name__ == "__main__":
    main()
