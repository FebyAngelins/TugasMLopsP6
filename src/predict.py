"""L8: panggil REST API hasil `mlflow models serve` (port 5001)."""
import requests

from data import X_test

payload = {"dataframe_split": X_test.head(3).to_dict(orient="split")}
r = requests.post("http://127.0.0.1:5001/invocations", json=payload)
print(r.json())  # {"predictions": [...]} (1 = malignant)
