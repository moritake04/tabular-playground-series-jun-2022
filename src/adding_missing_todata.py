import pandas as pd
from tqdm import tqdm


def main():
    # read csv
    data = pd.read_csv("../data/input/data_nan_flg.csv")
    imputed_data = pd.read_csv("../data/output/submission_seed44.csv")

    # adding predicted_missing_values to data
    for i in tqdm(range(len(imputed_data))):
        row_id, row_col = imputed_data["row-col"].iloc[i].split("-")
        data.loc[int(row_id), row_col] = imputed_data["value"].iloc[i]
    data.to_csv("../data/input/nn1st_seed44_added_data.csv", index=False)


if __name__ == "__main__":
    main()
