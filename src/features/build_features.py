import pandas as pd
import pathlib
import sys
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def edit_id_column(df):
    df['id'] = df['id'].str.replace('id', '', regex=False)
    df['id'] = df['id'].astype(int)

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine formula to compute distances between two sets of GPS coordinates.
    Returns distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers

    # Convert degrees to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["build_features"]

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = 'trip_duration'

    train_df = pd.read_csv(f"{input_path}/train.csv")
    test_df = pd.read_csv(f"{input_path}/test.csv")

    train_df = edit_id_column(train_df)
    test_df = edit_id_column(test_df)

    # Apply it to your dataframe
    train_df['trip_distance_km'] = haversine_np(
        train_df['pickup_latitude'],
        train_df['pickup_longitude'],
        train_df['dropoff_latitude'],
        train_df['dropoff_longitude']
    )

    # Apply it to your dataframe
    test_df['trip_distance_km'] = haversine_np(
        test_df['pickup_latitude'],
        test_df['pickup_longitude'],
        test_df['dropoff_latitude'],
        test_df['dropoff_longitude']
    )

    train_df.drop(columns={"pickup_datetime","dropoff_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"}, inplace=True)
    test_df.drop(columns={"pickup_datetime","dropoff_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"}, inplace=True)

    train_df["store_and_fwd_flag"] = train_df["store_and_fwd_flag"].map({"N": 1, "Y": 2})
    test_df["store_and_fwd_flag"] = test_df["store_and_fwd_flag"].map({"N": 1, "Y": 2})

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

if __name__ == "__main__":
    main()
