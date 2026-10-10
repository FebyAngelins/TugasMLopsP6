"""L6: evaluasi final di test set (sekali), pada kandidat terbaik hasil L5."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score)

from data import ROOT, X_test, y_test, test_df

runs = mlflow.search_runs(
    experiment_names=["bc-malignant-detection"],
    filter_string="tags.candidate = 'true' and metrics.cv_precision_mean >= 0.90",
    order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"],
)
best_run_id = runs.iloc[0]["run_id"]
best_model_uri = runs.iloc[0]["tags.model_uri"]
print("Evaluasi:", runs.iloc[0]["tags.mlflow.runName"], best_run_id)

model = mlflow.sklearn.load_model(best_model_uri)
y_pred = model.predict(X_test)
score = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
         else model.decision_function(X_test))

test_metrics = {
    "test_recall": recall_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_f1": f1_score(y_test, y_pred),
    "test_roc_auc": roc_auc_score(y_test, score),
}

with mlflow.start_run(run_id=best_run_id):  # tambahkan ke run yang SAMA
    mlflow.log_metrics(test_metrics)
    mlflow.log_input(mlflow.data.from_pandas(test_df, source="data/test.csv",
                     name="bc-test", targets="target"), context="testing")
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["benign", "malignant"])
    mlflow.log_figure(disp.figure_, "plots/confusion_matrix_test.png")
    plt.close(disp.figure_)
    mlflow.set_tag("test_evaluated", "true")

print(test_metrics)
# simpan untuk register.py
with open(os.path.join(ROOT, "data", "last_eval.json"), "w") as f:
    json.dump({"best_run_id": best_run_id, "best_model_uri": best_model_uri,
               "test_metrics": test_metrics}, f)
