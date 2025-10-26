# Image Classification Backend

FastAPI backend for serving PyTorch image classification models.

## Setup

### 1. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Add Your Model Weights
Update `main.py` to load your trained model weights:

\`\`\`python
# In the load_models() function, uncomment and update:
tiny_vgg.load_state_dict(torch.load("path/to/tiny_vgg.pth"))
googlenet.load_state_dict(torch.load("path/to/googlenet.pth"))
vit.load_state_dict(torch.load("path/to/vit.pth"))
\`\`\`

### 3. Run the Server
\`\`\`bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Endpoints

### GET /health
Health check endpoint

### GET /models
Get list of available models

### POST /predict
Single image prediction
- **Parameters**: 
  - `file`: Image file (multipart/form-data)
  - `model_name`: Model to use (TinyVGG, GoogleNet, ViT)

### POST /predict-batch
Batch prediction for multiple images

## Deployment Options

### Option 1: Heroku
\`\`\`bash
heroku create your-app-name
git push heroku main
\`\`\`

### Option 2: Railway
Connect your GitHub repo to Railway and deploy

### Option 3: AWS EC2
Deploy on an EC2 instance with GPU support

### Option 4: Google Cloud Run
\`\`\`bash
gcloud run deploy image-classifier --source .
\`\`\`

### Option 5: Modal (Recommended for ML)
\`\`\`bash
pip install modal
modal deploy main.py
\`\`\`

## Environment Variables

Update CORS origins in `main.py` for your frontend URL:
\`\`\`python
allow_origins=["http://localhost:3000", "https://your-frontend.com"]
\`\`\`

## GPU Support

The backend automatically uses GPU if available. Check with:
\`\`\`python
import torch
print(torch.cuda.is_available())
