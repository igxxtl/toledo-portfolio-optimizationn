from pathlib import Path
import sys

from gluonts.model.predictor import Predictor

project_root = Path(__file__).resolve().parent
lag_llama_dir = project_root / "lag-Llama"
model_dir = project_root / "modelo_7h_vwap"

if lag_llama_dir.exists():
    sys.path.insert(0, str(lag_llama_dir))

predictor = Predictor.deserialize(model_dir)

print("Tipo:", type(predictor))
print("prediction_length:", getattr(predictor, "prediction_length", None))
print("freq:", getattr(predictor, "freq", None))
print("lead_time:", getattr(predictor, "lead_time", None))
print("repr:", predictor)