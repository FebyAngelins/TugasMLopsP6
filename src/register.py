"""L7: registry + quality gate; L9: reproduce info dari champion."""
import json
import os

import mlflow
from mlflow import MlflowClient

from data import ROOT, data_md5

with open(os.path.join(ROOT, "data", "last_eval.json")) as f:
    ev = json.load(f)
best_model_uri, test_metrics = ev["best_model_uri"], ev["test_metrics"]

client, NAME = MlflowClient(), "bc-malignant-classifier"

mv = mlflow.register_model(best_model_uri, NAME)  # -> versi baru
client.update_model_version(
    NAME, mv.version,
    description=f"Dipilih otomatis: max cv_recall dgn precision>=0.90. data_md5={data_md5}")
client.set_model_version_tag(NAME, mv.version, "data_md5", data_md5)
client.set_registered_model_alias(NAME, "challenger", mv.version)


def champion_recall():
    try:
        champ = client.get_model_version_by_alias(NAME, "champion")
        return client.get_run(champ.run_id).data.metrics["test_recall"]
    except Exception:  # belum ada champion
        return -1.0


MIN_PRECISION, MARGIN = 0.90, 0.005
passed = (test_metrics["test_precision"] >= MIN_PRECISION and
          test_metrics["test_recall"] >= champion_recall() + MARGIN)

client.set_model_version_tag(NAME, mv.version, "validation_status",
                             "approved" if passed else "rejected")
if passed:
    client.set_registered_model_alias(NAME, "champion", mv.version)  # promosi
print(f"Versi {mv.version}: {'APPROVED -> @champion' if passed else 'REJECTED'}")

# ---------- L9: Reproduce ----------
champ = client.get_model_version_by_alias(NAME, "champion")
run = client.get_run(champ.run_id)
print("Run        :", run.info.run_name, run.info.run_id)
print("Git commit :", run.data.tags.get("mlflow.source.git.commit"))
print("Data md5   :", run.data.tags["data_md5"])
print("Params     :", {k: v for k, v in run.data.params.items() if k.startswith("clf__")})
print("Metrics    :", {k: round(v, 3) for k, v in run.data.metrics.items()})
assert data_md5 == run.data.tags["data_md5"], "Data berbeda -> hasil tidak akan sama!"
path = mlflow.artifacts.download_artifacts(run_id=champ.run_id, artifact_path="model")
print(open(f"{path}/requirements.txt").read())
