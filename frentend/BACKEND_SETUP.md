# Model Setup Guide

## Directory Structure

Your backend folder should look like this:

\`\`\`
backend/
├── main.py
├── requirements.txt
├── models/
│   ├── tiny_vgg.pth
│   ├── googlenet.pth
│   └── vit.pth
└── venv/
\`\`\`

## Steps to Set Up Your Trained Models

### 1. Create Models Directory
\`\`\`bash
cd backend
mkdir models
\`\`\`

### 2. Upload Your Trained Models

From Google Colab, download your trained model files and place them in the `backend/models/` directory:

- **TinyVGG**: `tiny_vgg.pth`
- **GoogleNet**: `googlenet.pth`
- **Vision Transformer**: `vit.pth`

### 3. Verify Model Files

\`\`\`bash
ls -la models/
\`\`\`

You should see:
\`\`\`
tiny_vgg.pth
googlenet.pth
vit.pth
\`\`\`

### 4. Run the Backend

\`\`\`bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Run the server
uvicorn main:app --reload --port 8000
\`\`\`

### 5. Check Model Loading

Open http://localhost:8000/docs and check the logs. You should see:
\`\`\`
✓ TinyVGG loaded from backend/models/tiny_vgg.pth
✓ GoogleNet loaded from backend/models/googlenet.pth
✓ Vision Transformer loaded from backend/models/vit.pth
\`\`\`

## Troubleshooting

### Models Not Loading?

1. **Check file names**: Make sure they match exactly:
   - `tiny_vgg.pth` (not `TinyVGG.pth` or `tiny-vgg.pth`)
   - `googlenet.pth`
   - `vit.pth`

2. **Check file location**: Files must be in `backend/models/` directory

3. **Check file format**: Models must be saved as `.pth` files

4. **Check model architecture**: Ensure your saved models match the architecture defined in `main.py`

### Model Architecture Mismatch?

If you get an error like "unexpected key in state_dict", your saved model might have a different architecture. You can:

1. Update the model class definitions in `main.py` to match your Colab implementation
2. Or save your models with `torch.save(model, 'model.pth')` instead of `torch.save(model.state_dict(), 'model.pth')`

## Testing the API

Once running, test with:

\`\`\`bash
curl -X POST "http://localhost:8000/predict?model_name=TinyVGG" \
  -H "accept: application/json" \
  -F "file=@path/to/your/image.jpg"
\`\`\`

Or use the interactive API docs at http://localhost:8000/docs
