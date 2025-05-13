import pandas as pd
import pathlib
import sys
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def edit_id_column(df):
    df['id'] = df['id'].str.replace('id', '', regex=False)
    df['id'] = df['id'].astype(int)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["build_features"]

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = 'Class'

    train_df = pd.read_csv(f"{input_path}/train.csv")
    test_df = pd.read_csv(f"{input_path}/test.csv")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    train_df = edit_id_column(train_df)
    test_df = edit_id_column(test_df)

if __name__ == "__main__":
    main()
