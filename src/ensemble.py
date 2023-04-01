import pandas as pd
from tqdm import tqdm


def main():
    # read csv
    sample_submission = pd.read_csv("../data/output/sample_submission.csv")
    sub1_1 = pd.read_csv("../data/output/submission_seed42.csv")
    sub1_2 = pd.read_csv("../data/output/submission2_seed42.csv")
    sub2_1 = pd.read_csv("../data/output/submission_seed43.csv")
    sub2_2 = pd.read_csv("../data/output/submission2_seed43.csv")
    sub3_1 = pd.read_csv("../data/output/submission_seed44.csv")
    sub3_2 = pd.read_csv("../data/output/submission2_seed44.csv")

    # ensemble
    for i in tqdm(range(len(sample_submission))):
        sub1 = (sub1_1.loc[i, "value"] + sub1_2.loc[i, "value"]) / 2.0
        sub2 = (sub2_1.loc[i, "value"] + sub2_2.loc[i, "value"]) / 2.0
        sub3 = (sub3_1.loc[i, "value"] + sub3_2.loc[i, "value"]) / 2.0
        imputed_value = (sub1 + sub2 + sub3) / 3.0
        sample_submission.loc[i, "value"] = imputed_value
    sample_submission.to_csv(
        "../data/output/submission_1st2nd_3seed_ensemble.csv", index=False
    )


if __name__ == "__main__":
    main()
