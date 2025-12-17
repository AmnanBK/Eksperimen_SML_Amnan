import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Fungsi untuk load data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Data berhasil dimuat dari {path}.")
    return df


# Fungsi untuk preprocessing data
def preprocess_data(df):
    target_col = "Potability"
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cols = X.columns

    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_final = pd.DataFrame(X_train_scaled, columns=cols)
    X_test_final = pd.DataFrame(X_test_scaled, columns=cols)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("[INFO] Preprocessing data telah selesai.")
    return X_train_final, X_test_final, y_train, y_test


# Menyimpan data hasil clean ke CSV
def save_data(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(output_dir, "train_potability.csv")
    test_path = os.path.join(output_dir, "test_potability.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"[INFO] Data train disimpan di: {train_path}")
    print(f"[INFO] Data test disimpan di: {test_path}")


# Fungsi utama
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(
        base_dir, "..", "water_potability_raw", "water_potability.csv"
    )
    output_dir = os.path.join(base_dir, ".." "water_potability_preprocessing")

    print("[INFO] Memulai Otomatisasi Preprocessing")
    df = load_data(raw_data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    save_data(X_train, X_test, y_train, y_test, output_dir)
    print("[INFO] Semua proses telah selesai")


if __name__ == "__main__":
    main()
