# server.py
import io, os, traceback, json, inspect
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from PIL import Image
import app as core
from math import isfinite

MALIGNANT = {"mel", "bcc", "akiec", "scc"}

app = FastAPI(title="HealthAware ML Service", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        return f if isfinite(f) else default
    except Exception:
        return default

def _class_info() -> Dict[str, Dict[str, Any]]:
    ci = getattr(core, "CLASS_INFO", None)
    return ci if isinstance(ci, dict) else {}

def _compute_risk(class_name: str, confidence: float) -> str:
    if class_name.lower() in MALIGNANT:
        if confidence >= 0.6: return "High"
        if confidence >= 0.35: return "Medium"
        return "Low"
    if confidence >= 0.8: return "Low"
    if confidence >= 0.5: return "Medium"
    return "Medium"

def _normalize_prediction(raw: Any) -> Dict[str, Any]:
    ci = _class_info()

    if isinstance(raw, dict) and "prediction" in raw and "class_probabilities" in raw:
        pred = raw["prediction"]
        display_name = pred.get("display_name") or ci.get(pred.get("class_name"), {}).get("display", pred.get("class_name", "Unknown"))
        risk = pred.get("risk") or _compute_risk(pred.get("class_name", ""), _safe_float(pred.get("confidence", 0)))
        desc = pred.get("description") or ci.get(pred.get("class_name"), {}).get("description", "")
        reco = pred.get("recommendation") or ci.get(pred.get("class_name"), {}).get("recommendation", "")
        pred.update({
            "display_name": display_name,
            "risk": risk,
            "description": desc,
            "recommendation": reco,
        })
        raw["prediction"] = pred
        raw.setdefault("model_predictions", {})
        return raw

    class_probs = {}
    top_class = None
    top_conf = 0.0

    if isinstance(raw, dict):
        if "class_probabilities" in raw and isinstance(raw["class_probabilities"], dict):
            class_probs = raw["class_probabilities"]
        elif "probs" in raw and isinstance(raw["probs"], dict):
            class_probs = raw["probs"]

        if "top" in raw and isinstance(raw["top"], dict):
            top_class = raw["top"].get("class_name") or raw["top"].get("label") or raw["top"].get("class")
            top_conf = _safe_float(raw["top"].get("confidence") or raw["top"].get("score") or 0)
        elif "pred" in raw and isinstance(raw["pred"], (list, tuple)) and raw["pred"]:
            cand = raw["pred"][0]
            if isinstance(cand, (list, tuple)) and len(cand) >= 2:
                top_class, top_conf = cand[0], _safe_float(cand[1])
        elif "class_name" in raw and "confidence" in raw:
            top_class = raw["class_name"]
            top_conf = _safe_float(raw["confidence"])

    if class_probs and (top_class is None):
        top_class, top_conf = max(class_probs.items(), key=lambda kv: kv[1])

    top_class = (top_class or "unknown").lower()
    top_conf = _safe_float(top_conf)

    model_predictions: Dict[str, Dict[str, Any]] = raw.get("model_predictions", {}) if isinstance(raw, dict) else {}

    class_id = -1
    if hasattr(core, "CLASS_NAMES") and isinstance(core.CLASS_NAMES, dict):
        keys = list(core.CLASS_NAMES.keys())
        if top_class in keys:
            class_id = keys.index(top_class)

    display_name = ci.get(top_class, {}).get("display", top_class.title())
    desc = ci.get(top_class, {}).get("description", "")
    reco = ci.get(top_class, {}).get("recommendation", "")
    risk = _compute_risk(top_class, top_conf)

    return {
        "prediction": {
            "class_id": class_id,
            "class_name": top_class,
            "display_name": display_name,
            "risk": risk,
            "description": desc,
            "recommendation": reco,
            "confidence": top_conf,
        },
        "model_predictions": model_predictions,
        "class_probabilities": class_probs,
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    userId: Optional[str] = Form(default=None),  # optional personalization
    meta: Optional[str] = Form(default=None),    # optional JSON string
):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    meta_obj: Optional[dict] = None
    if meta:
        try:
            meta_obj = json.loads(meta)
        except Exception:
            meta_obj = None

    try:
        raw = None

        # 1) Prefer predict_image(image, user_id=..., meta=...)
        if hasattr(core, "predict_image") and callable(core.predict_image):
            raw = core.predict_image(image, user_id=userId, meta=meta_obj)

        # 2) Else try inference(image, ...) with kwargs only if accepted
        elif hasattr(core, "inference") and callable(core.inference):
            fn = core.inference
            sig = inspect.signature(fn)
            kwargs = {}
            if "user_id" in sig.parameters: kwargs["user_id"] = userId
            if "meta" in sig.parameters: kwargs["meta"] = meta_obj
            raw = fn(image, **kwargs) if "image" in sig.parameters else fn(**kwargs)

        # 3) Else fallback to predict(...) WITHOUT user kwargs (to avoid your error)
        elif hasattr(core, "predict") and callable(core.predict):
            fn = core.predict
            sig = inspect.signature(fn)
            # if your predict accepts a PIL image param named 'image', pass it; otherwise just call and let your code load from disk
            if "image" in sig.parameters:
                raw = fn(image)
            else:
                # Last resort: call with no kwargs; your function likely loads internally
                raw = fn()

        else:
            raise RuntimeError("No predict_image() / inference() / predict() function found in app.py")

        normalized = _normalize_prediction(raw)
        return jsonable_encoder(normalized)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}
