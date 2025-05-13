import os
import traceback
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── 0. Helpers ────────────────────────────────────────────────────────────
def load_model() -> joblib:
    """
    Try to load one of our model pickles from the project root.
    """
    candidates = ["propeller_model_pipeline.pkl", "gbr_propeller_model.pkl"]
    cwd = os.getcwd()
    for fname in candidates:
        path = os.path.join(cwd, fname)
        if os.path.isfile(path):
            print(f"✅ Loading model from: {path}", flush=True)
            return joblib.load(path)
    raise FileNotFoundError(f"None of {candidates} found in {cwd}")

# ─── 1. Input schema ────────────────────────────────────────────────────────
class PropellerInput(BaseModel):
    blade_loading: float
    cp_log: float
    j2: float
    solidity: float

# ─── 2. App & model startup ─────────────────────────────────────────────────
app = FastAPI()
print("Working directory:", os.getcwd(), flush=True)

try:
    model = load_model()
except Exception as e:
    print("❌ Model loading failed:", e, flush=True)
    traceback.print_exc()
    # Prevent the server from starting without a model
    raise

# ─── 3. Health check ────────────────────────────────────────────────────────
@app.get("/", response_model=dict)
def read_root():
    return {"message": "Propeller prediction API is up and running!"}

# ─── 4. Prediction endpoint ─────────────────────────────────────────────────
@app.post("/predict", response_model=dict)
async def predict(data: PropellerInput):
    """
    Accepts JSON:
    {
      "blade_loading": float,
      "cp_log":       float,
      "j2":           float,
      "solidity":     float
    }
    Returns JSON:
    {
      "predicted_ct": float
    }
    """
    print("🟢 Received input:", data, flush=True)
    features = [[
        data.blade_loading,
        data.cp_log,
        data.j2,
        data.solidity
    ]]
    print("🟢 Feature vector:", features, flush=True)

    try:
        pred = model.predict(features)
        print("🟢 Raw prediction:", pred, flush=True)
        return {"predicted_ct": float(pred[0])}
    except Exception as e:
        print("🔴 Prediction error:", e, flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
