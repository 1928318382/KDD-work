import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

NORMAL_FILE = "normal.csv"
ATTACK_FILE = "attack.csv"
MERGED_FILE = "merged.csv"

OUT_NORMAL = "swat_clean_normal.csv"
OUT_ATTACK = "swat_clean_attack.csv"
OUT_MERGED = "swat_clean_merged.csv"
OUT_ALL = "swat_clean_all.csv"

ROLLING_WINDOW = 5

COL_TIME = "Timestamp"
COL_STATE = "Normal/Attack"
COL_LABEL = "label"


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def add_label(df):
    df[COL_LABEL] = (df[COL_STATE] != "Normal").astype(int)
    return df


def drop_unused(df):
    drop_cols = []
    if COL_TIME in df.columns:
        drop_cols.append(COL_TIME)
    if COL_STATE in df.columns:
        drop_cols.append(COL_STATE)
    return df.drop(columns=drop_cols)


def split_sensor_actuator(normal_df):
    feature_cols = [c for c in normal_df.columns if c != COL_LABEL]
    unique_counts = normal_df[feature_cols].nunique()
    actuator_cols = unique_counts[unique_counts <= 3].index.tolist()
    sensor_cols = unique_counts[unique_counts > 3].index.tolist()
    if len(sensor_cols) == 0:
        raise ValueError("No sensor columns found.")
    return sensor_cols, actuator_cols


def fill_missing(df, sensor_cols):
    df[sensor_cols] = df[sensor_cols].interpolate(method="linear", limit_direction="both")
    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    return df


def denoise(df, sensor_cols, window):
    df[sensor_cols] = df[sensor_cols].rolling(window=window, center=True).mean()
    df[sensor_cols] = df[sensor_cols].bfill().ffill()
    return df


def standardize(normal_df, attack_df, merged_df, sensor_cols):
    scaler = StandardScaler()
    normal_df[sensor_cols] = scaler.fit_transform(normal_df[sensor_cols])
    attack_df[sensor_cols] = scaler.transform(attack_df[sensor_cols])
    merged_df[sensor_cols] = scaler.transform(merged_df[sensor_cols])
    return normal_df, attack_df, merged_df


def check_nan(df, sensor_cols, name):
    if df[sensor_cols].isna().sum().sum() != 0:
        raise ValueError(f"{name} still has NaN values.")


def main():
    normal = load_csv(NORMAL_FILE)
    attack = load_csv(ATTACK_FILE)
    merged = load_csv(MERGED_FILE)

    normal = add_label(normal)
    attack = add_label(attack)
    merged = add_label(merged)

    normal = drop_unused(normal)
    attack = drop_unused(attack)
    merged = drop_unused(merged)

    sensor_cols, actuator_cols = split_sensor_actuator(normal)

    normal = fill_missing(normal, sensor_cols)
    attack = fill_missing(attack, sensor_cols)
    merged = fill_missing(merged, sensor_cols)

    check_nan(normal, sensor_cols, "normal")
    check_nan(attack, sensor_cols, "attack")
    check_nan(merged, sensor_cols, "merged")

    normal = denoise(normal, sensor_cols, ROLLING_WINDOW)
    attack = denoise(attack, sensor_cols, ROLLING_WINDOW)
    merged = denoise(merged, sensor_cols, ROLLING_WINDOW)

    normal, attack, merged = standardize(normal, attack, merged, sensor_cols)

    normal.to_csv(OUT_NORMAL, index=False)
    attack.to_csv(OUT_ATTACK, index=False)
    merged.to_csv(OUT_MERGED, index=False)

    all_df = pd.concat([normal, attack], ignore_index=True)
    all_df.to_csv(OUT_ALL, index=False)

    print("Preprocessing finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
