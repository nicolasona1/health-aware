import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as torchvision_models  # Renamed to avoid conflict
from collections import Counter
import json
import warnings
import argparse
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Define class names and info
CLASS_NAMES = {
    0: 'akiec',  # Actinic Keratoses and Intraepithelial Carcinoma
    1: 'bcc',    # Basal Cell Carcinoma
    2: 'bkl',    # Benign Keratosis-like Lesions
    3: 'df',     # Dermatofibroma
    4: 'mel',    # Melanoma
    5: 'nv',     # Melanocytic Nevi
    6: 'vasc'    # Vascular Lesions
}

CLASS_INFO = {
    'akiec': {
        'display': 'Actinic Keratosis / Intraepithelial Carcinoma',
        'risk': 'High',
        'description': 'A pre-cancerous growth or early form of skin cancer.',
        'recommendation': 'Consult a dermatologist promptly for evaluation.'
    },
    'bcc': {
        'display': 'Basal Cell Carcinoma',
        'risk': 'High',
        'description': 'The most common type of skin cancer, usually slow-growing.',
        'recommendation': 'Consult a dermatologist for proper treatment options.'
    },
    'bkl': {
        'display': 'Benign Keratosis',
        'risk': 'Low',
        'description': 'A non-cancerous skin growth that appears as a waxy, scaly patch.',
        'recommendation': 'Generally no treatment needed, but monitor for changes.'
    },
    'df': {
        'display': 'Dermatofibroma',
        'risk': 'Low',
        'description': 'A common benign skin growth or nodule that is usually harmless.',
        'recommendation': 'No treatment needed unless causing discomfort.'
    },
    'mel': {
        'display': 'Melanoma',
        'risk': 'High',
        'description': 'A serious form of skin cancer that can spread if not treated early.',
        'recommendation': 'Seek immediate medical attention.'
    },
    'nv': {
        'display': 'Melanocytic Nevus (Mole)',
        'risk': 'Low',
        'description': 'A benign growth of melanocytes, usually harmless but should be monitored.',
        'recommendation': 'Monitor for changes in size, shape, or color.'
    },
    'vasc': {
        'display': 'Vascular Lesion',
        'risk': 'Low',
        'description': 'Abnormalities of blood vessels that appear on the skin surface.',
        'recommendation': 'Typically harmless but consult a doctor if concerned.'
    }
}

def load_models(models_dir="./models"):
    """Load all models from the models directory"""
    loaded_models = {}  # Renamed from 'models' to avoid conflict with the imported module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} not found. Searching for models...")
        # Try to find models in parent directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, files in os.walk(os.path.dirname(script_dir)):
            if "models" in dirs:
                models_dir = os.path.join(root, "models")
                print(f"Found models directory at: {models_dir}")
                break
    
    # Possible model filenames (with variations)
    possible_models = {
        "mobilenetv3": ["final_mobilenetv3_model.pt", "mobilenetv3.pt", "final_mobilenetv3.pt"],
        "densenet121": ["final_densenet121_model.pt", "densenet121.pt", "final_densenet121.pt"],
        "resnet50": ["final_resnet50_model.pt", "resnet50.pt", "final_resnet50.pt"]
    }
    
    # Find and load each model
    for model_name, filenames in possible_models.items():
        model_found = False
        for filename in filenames:
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                print(f"Loading {model_name} model from {model_path}")
                try:
                    # Initialize the base model architecture first
                    if 'resnet' in model_name:
                        model = torchvision_models.resnet50(pretrained=False)
                        model.fc = torch.nn.Linear(model.fc.in_features, 7)
                    elif 'densenet' in model_name:
                        model = torchvision_models.densenet121(pretrained=False)
                        model.classifier = torch.nn.Linear(model.classifier.in_features, 7)
                    elif 'mobilenet' in model_name:
                        model = torchvision_models.mobilenet_v3_large(pretrained=False)
                        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 7)
                    else:
                        raise ValueError(f"Unknown model architecture: {model_name}")
                    
                    # Load model to the appropriate device
                    model_data = torch.load(model_path, map_location=device)
                    
                    # Handle different saved formats
                    if isinstance(model_data, dict) and 'model' in model_data:
                        # If the file contains a dict with 'model' key
                        model.load_state_dict(model_data['model'])
                    elif isinstance(model_data, dict) and 'state_dict' in model_data:
                        # If the file contains a state_dict
                        model.load_state_dict(model_data['state_dict'])
                    elif isinstance(model_data, dict) and len(model_data) > 0:
                        # If the file contains just a state dict (OrderedDict)
                        # Clean state_dict if needed (remove module. prefix)
                        cleaned_state_dict = {k.replace('module.', ''): v for k, v in model_data.items()}
                        model.load_state_dict(cleaned_state_dict, strict=False)
                    elif hasattr(model_data, 'state_dict'):
                        # If the file contains the model object directly
                        model.load_state_dict(model_data.state_dict())
                    
                    model.eval()  # Set model to evaluation mode
                    model = model.to(device)  # Ensure model is on the correct device
                    loaded_models[model_name] = model
                    print(f"Successfully loaded {model_name}")
                    model_found = True
                    break
                except Exception as e:
                    print(f"Error loading {model_name} from {model_path}: {e}")
        
        if not model_found:
            print(f"Warning: Could not find or load any {model_name} model files!")
    
    # Check if any models were loaded
    if not loaded_models:
        print("No models could be loaded! Unable to make predictions.")
        return None, device
    else:
        print(f"Loaded {len(loaded_models)} models successfully")
        
    return loaded_models, device

def preprocess_image(image_path, device):
    """Process image for model input"""
    try:
        # Define image transformations
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Open image from file
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations and prepare tensor
        tensor = image_transforms(image).unsqueeze(0).to(device)
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict(image_path, loaded_models, device):
    """Predict skin lesion from image file (ensemble = mean of probabilities)."""
    if not loaded_models:
        print("Error: No models loaded for prediction")
        return None

    # Preprocess image
    image_tensor = preprocess_image(image_path, device)
    if image_tensor is None:
        return None

    predictions = {}                       # per-backbone predictions
    num_classes = 7
    sum_probs = np.zeros(num_classes, dtype=np.float32)

    with torch.no_grad():
        for model_name, model in loaded_models.items():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()  # shape [7]
            pred_idx = int(probs.argmax())

            predictions[model_name] = {
                "class_id": pred_idx,
                "class_name": CLASS_NAMES[pred_idx],
                "confidence": float(probs[pred_idx]),   # 0..1 for this backbone
            }

            sum_probs += probs

    # ---- Ensemble by averaging probabilities (not voting) ----
    n_models = max(1, len(loaded_models))
    ens_probs = (sum_probs / n_models).astype(float)            # shape [7], sums ~1
    top_idx = int(ens_probs.argmax())
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(ens_probs[top_idx])                        # 0..1

    # (Optional) agreement: fraction of models that chose the ensemble top class
    agreement = sum(1 for v in predictions.values() if v["class_id"] == top_idx) / n_models

    class_probs = {CLASS_NAMES[i]: float(ens_probs[i]) for i in range(num_classes)}
    info = CLASS_INFO.get(top_label, {})

    result = {
        "prediction": {
            "class_id": top_idx,
            "class_name": top_label,
            "display_name": info.get("display", top_label),
            "risk": info.get("risk", "Low"),
            "description": info.get("description", ""),
            "recommendation": info.get("recommendation", ""),
            "confidence": top_prob,        # <-- now TRUE confidence from ensemble (0..1)
            "agreement": agreement,        # <-- optional, shows model consensus (0..1)
        },
        "model_predictions": predictions,  # per-backbone outputs
        "class_probabilities": class_probs # ensemble distribution used for the bars
    }
    return result

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Please Input the Name of your File: ')
    parser.add_argument('-f', '--file', help='Path to input file')
    parser.add_argument('--models', type=str, help='Path to the models directory (default: auto-detect)')
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define models directory path
    models_dir = args.models if args.models else os.environ.get("MODELS_DIR", os.path.join(script_dir, "models"))
    
    # Process the file argument
    if args.file:
        print(f'Processing file: {args.file}')
        image_path = args.file
    else:
        print('File not specified')
        # Default image path if file not specified
        image_path = os.path.join(script_dir, "melanoma.jpeg")
        
    print(f"Processing image: {image_path}")
    print(f"Script directory: {script_dir}")
    print(f"Looking for models in: {models_dir}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        # Try to find the image in the parent directory structure
        print("Image not found at the expected location. Searching...")
        for root, dirs, files in os.walk(os.path.dirname(script_dir)):
            image_filename = os.path.basename(image_path)
            if image_filename in files:
                image_path = os.path.join(root, image_filename)
                print(f"Found image at: {image_path}")
                break
        
        # If still not found
        if not os.path.exists(image_path):
            print(f"Error: Could not find image {image_path} anywhere!")
            return
    
    # Load models and make prediction
    loaded_models, device = load_models(models_dir)
    if not loaded_models:
        print("Error: Could not load any models. Exiting.")
        return
        
    result = predict(image_path, loaded_models, device)
    if not result:
        print("Error: Could not make prediction. Exiting.")
        return
    
    # Print results
    print("\n=== PREDICTION RESULTS ===")
    print(f"Classification: {result['prediction']['display_name']}")
    print(f"Risk Level: {result['prediction']['risk']}")
    print(f"Description: {result['prediction']['description']}")
    print(f"Recommendation: {result['prediction']['recommendation']}")
    print(f"Confidence: {result['prediction']['confidence'] * 100:.2f}%\n")
    
    print("Individual Model Predictions:")
    for model_name, pred in result['model_predictions'].items():
        print(f"  - {model_name}: {CLASS_INFO[pred['class_name']]['display']} (Confidence: {pred['confidence'] * 100:.2f}%)")
    
    print("\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  - {CLASS_INFO[class_name]['display']}: {prob * 100:.2f}%")
    
    # Save results to JSON file
    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_prediction.json"
    with open(output_filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults also saved to {output_filename}")

# ==== FASTAPI INTEGRATION WRAPPER (add this) ====
# CHANGE: Keep models in memory for repeated API calls
from PIL import Image as _PilImage
import tempfile as _temp
import os as _os

# Global cache for models/device
_LOADED_MODELS = None
_DEVICE = None

def _ensure_loaded():
    """Load models once and reuse (idempotent)."""
    global _LOADED_MODELS, _DEVICE
    if _LOADED_MODELS is None:
        # CHANGE: allow MODELS_DIR override via env; default to ./models next to this file
        script_dir = _os.path.dirname(_os.path.abspath(__file__))
        models_dir = _os.environ.get("MODELS_DIR", _os.path.join(script_dir, "models"))
        _LOADED_MODELS, _DEVICE = load_models(models_dir=models_dir)

def predict_image(image: _PilImage.Image, user_id=None, meta=None):
    """
    CHANGE: FastAPI will call this version with a PIL Image.
    We save to a temp file and reuse your existing `predict(image_path, loaded_models, device)`.
    """
    _ensure_loaded()
    if _LOADED_MODELS is None:
        raise RuntimeError("Models failed to load; check MODELS_DIR and weight files.")

    # Save PIL image to a temporary file so we can reuse your preprocess path
    tmp_path = None
    try:
        with _temp.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_rgb = image.convert("RGB")
            image_rgb.save(tmp.name, format="JPEG", quality=95)
            tmp_path = tmp.name

        # Call your original predictor (ensemble) – returns the final dict your UI expects
        result = predict(tmp_path, _LOADED_MODELS, _DEVICE)

        # (Optional) attach passthrough info
        if isinstance(result, dict):
            if user_id is not None:
                result["user_id"] = user_id
            if meta is not None:
                result["meta"] = meta

        return result
    finally:
        # Clean up the temp file
        if tmp_path:
            try:
                _os.remove(tmp_path)
            except Exception:
                pass
# ==== END WRAPPER ====
if __name__ == "__main__":
    main()