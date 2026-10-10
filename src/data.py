"""L1: load, split tetap, dan hash versi data."""
import hashlib
import os

import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

EXPERIMENT = "bc-malignant-detection"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "data", "train.csv")
TEST_CSV = os.path.join(ROOT, "data", "test.csv")

mlflow.set_experiment(EXPERIMENT)

df = load_breast_cancer(as_frame=True).frame
df["target"] = 1 - df["target"]  # 1 = malignant (kelas positif)

# Sidik jari data: berubah satu nilai pun -> hash berbeda
data_md5 = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["target"], random_state=42
)
os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)  # test set: DIKUNCI sampai L6

X_train, y_train = train_df.drop(columns="target"), train_df["target"]
X_test, y_test = test_df.drop(columns="target"), test_df["target"]

# Objek Dataset MLflow -> tampil di tab "Datasets" pada setiap run
train_ds = mlflow.data.from_pandas(
    train_df, source="data/train.csv", name="bc-train", targets="target"
)
COMMON_TAGS = {
    "data_md5": data_md5,
    "split_seed": "42",
    "dataset": "sklearn-breast-cancer",
}
