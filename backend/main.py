
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
from typing import Dict, List
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Classification API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Class names for your models
CLASS_NAMES = ["pizza", "steak", "sushi"]

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============ MODEL DEFINITIONS ============

class TinyVGG(nn.Module):
    """TinyVGG model - matches your training implementation exactly"""
    def __init__(self, input_shape: int = 3, hidden_units: int = 10, output_shape: int = 3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=3, 
                 embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = x.reshape(B, 3, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3).reshape(B, -1, 3 * self.patch_size * self.patch_size)
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x


# ============ MODEL LOADING ============

models_dict: Dict[str, nn.Module] = {}

def load_models():
    """Load all models from the models directory"""
    global models_dict
    try:
        logger.info("Loading TinyVGG model...")
        tiny_vgg = TinyVGG(input_shape=3, hidden_units=10, output_shape=3).to(device)
        tiny_vgg_path = MODELS_DIR / "tiny_vgg.pth"
        if tiny_vgg_path.exists():
            tiny_vgg.load_state_dict(torch.load(tiny_vgg_path, map_location=device))
            logger.info(f"✓ TinyVGG loaded from {tiny_vgg_path}")
        else:
            logger.warning(f"⚠ TinyVGG weights not found at {tiny_vgg_path}")
        models_dict["TinyVGG"] = tiny_vgg.eval()
        
        logger.info("Loading GoogleNet model...")
        googlenet = models.googlenet(pretrained=False).to(device)
        googlenet.fc = nn.Linear(1024, 3)
        googlenet_path = MODELS_DIR / "googlenet.pth"
        if googlenet_path.exists():
            googlenet.load_state_dict(torch.load(googlenet_path, map_location=device), strict=False)
            logger.info(f"✓ GoogleNet loaded from {googlenet_path}")
        else:
            logger.warning(f"⚠ GoogleNet weights not found at {googlenet_path}")
        models_dict["GoogleNet"] = googlenet.eval()
        
        logger.info("Loading Vision Transformer model...")
        vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12).to(device)
        vit_path = MODELS_DIR / "vit.pth"
        if vit_path.exists():
            vit.load_state_dict(torch.load(vit_path, map_location=device), strict=False)
            logger.info(f"✓ Vision Transformer loaded from {vit_path}")
        else:
            logger.warning(f"⚠ Vision Transformer weights not found at {vit_path}")
        models_dict["ViT"] = vit.eval()
        
        logger.info("All models initialized successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Load models on startup
load_models()


# ============ API ENDPOINTS ============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": list(models_dict.keys())
    }


@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            {
                "id": "TinyVGG",
                "name": "TinyVGG",
                "description": "Lightweight CNN model optimized for speed"
            },
            {
                "id": "GoogleNet",
                "name": "GoogleNet",
                "description": "Transfer learning model with pre-trained weights"
            },
            {
                "id": "ViT",
                "name": "Vision Transformer",
                "description": "State-of-the-art transformer-based model"
            }
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Query("GoogleNet")):
    """
    Predict image classification
    
    Args:
        file: Image file to classify
        model: Name of the model to use (tinyvgg, googlenet, or vit)
    
    Returns:
        Prediction results with confidence scores
    """
    try:
        model_mapping = {
            "tinyvgg": "TinyVGG",
            "googlenet": "GoogleNet",
            "vit": "ViT"
        }
               # Normalize model name
        model_lower = model.lower().strip()
        logger.info(f"[DEBUG] Received model parameter: '{model}'")
        logger.info(f"[DEBUG] Model after normalization: '{model_lower}'")
        
        model_key = model_mapping.get(model_lower)
        logger.info(f"[DEBUG] Model key from mapping: '{model_key}'")
        logger.info(f"[DEBUG] Available models: {list(models_dict.keys())}")


        if not model_key:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' not found. Available models: tinyvgg, googlenet, vit"
            )
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"model_key is {model_key}")
        if model_key == "TinyVGG":
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        model_obj = models_dict[model_key]
        with torch.no_grad():
            logits = model_obj(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # Process results
        probs = probabilities[0].cpu().numpy()
        predicted_class_idx = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probs[predicted_class_idx])
        
        # Create detailed results
        predictions = [
            {
                "class": CLASS_NAMES[i],
                "confidence": float(probs[i]),
                "percentage": float(probs[i] * 100)
            }
            for i in range(len(CLASS_NAMES))
        ]
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "model": model_key,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": confidence * 100,
            "all_predictions": predictions,
            "image_name": file.filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...), model_name: str = "TinyVGG"):
    """Batch prediction for multiple images"""
    results = []
    for file in files:
        result = await predict(file, model_name)
        results.append(result)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

