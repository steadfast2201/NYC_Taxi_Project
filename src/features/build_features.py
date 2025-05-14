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
    return df

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

def modify_datetime(df):

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # Compute trip duration in minutes
    df["trip_time_minutes"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60

    return df

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

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    X_train = edit_id_column(X_train)
    X_test = edit_id_column(X_test)

    # Apply it to your dataframe
    X_train['trip_distance_km'] = haversine_np(
        X_train['pickup_latitude'],
        X_train['pickup_longitude'],
        X_train['dropoff_latitude'],
        X_train['dropoff_longitude']
    )

    # Apply it to your dataframe
    X_test['trip_distance_km'] = haversine_np(
        X_test['pickup_latitude'],
        X_test['pickup_longitude'],
        X_test['dropoff_latitude'],
        X_test['dropoff_longitude']
    )


    X_train["store_and_fwd_flag"] = X_train["store_and_fwd_flag"].map({"N": 1, "Y": 2})
    X_test["store_and_fwd_flag"] = X_test["store_and_fwd_flag"].map({"N": 1, "Y": 2})

    X_train = modify_datetime(X_train)
    X_test = modify_datetime(X_test)

    X_train.drop(columns={"pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","pickup_datetime", "dropoff_datetime"}, inplace=True)
    X_test.drop(columns={"pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","pickup_datetime", "dropoff_datetime"}, inplace=True)
    
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test


    train_df.to_csv(f"{output_path}/train.csv", index=False)
    test_df.to_csv(f"{output_path}/test.csv", index=False)

if __name__ == "__main__":
    main()
