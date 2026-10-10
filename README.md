# bc-mlops  MLflow Hands-on 2 (Breast Cancer Malignant Detection)

Alur eksperimen end-to-end dengan MLflow + scikit-learn: tracking, tuning (nested runs),
evaluasi test set, Model Registry dengan quality gate, dan serving REST API.

## Struktur
| File | Fungsi |
|---|---|
| `src/data.py` | Load, split 80/20, hash MD5 data |
| `src/tracking.py` | Fungsi logging standar `run_experiment()` |
| `src/train.py` | Baseline, seleksi model, GridSearch SVC, pemilihan kandidat |
| `src/evaluate.py` | Evaluasi sekali di test set |
| `src/register.py` | Registry, quality gate `@champion`, reproduce |
| `src/predict.py` | Klien REST API |

## Cara menjalankan
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000

# Terminal 2
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train.py
python src/evaluate.py
python src/register.py

# Serving (Terminal 3), lalu uji (Terminal 4)
mlflow models serve -m "models:/bc-malignant-classifier@champion" -p 5001 --env-manager local
python src/predict.py
```
Buka UI di http://127.0.0.1:5000 (mode *Model training*).

## Hasil utama
- Model terpilih: SVC (C=10, gamma=scale) — CV recall 0.965, precision 0.971
- Test set: recall 0.929, precision 1.000, ROC-AUC 0.993
