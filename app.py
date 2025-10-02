from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix PyTorch loading issue
torch.serialization.add_safe_globals(["ultralytics.nn.tasks.SegmentationModel"])
model = YOLO('best.pt')

@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...), confidence: float = 0.3):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        img_array = np.array(image)
        
        results = model(img_array, conf=confidence, save=False)
        annotated = results[0].plot(line_width=3, font_size=1.5)
        
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        num_potholes = len(results[0].boxes) if results[0].boxes is not None else 0
        
        return {
            "success": True,
            "num_potholes": num_potholes,
            "output_image": f"data:image/jpeg;base64,{img_base64}",
            "message": f"Found {num_potholes} pothole(s)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Pothole Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area.dragover { border-color: #007bff; background: #f8f9fa; }
        .result { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .result img { max-width: 100%; height: auto; }
        .loading { display: none; text-align: center; }
        .confidence-slider { margin: 20px 0; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
    </style>
</head>
<body>
    <h1>🕳️ Pothole Detection System</h1>
    
    <div class="confidence-slider">
        <label for="confidence">Confidence: <span id="conf-value">0.3</span></label>
        <input type="range" id="confidence" min="0.1" max="0.9" step="0.1" value="0.3">
    </div>

    <div class="upload-area" id="uploadArea">
        <p>📸 Upload an image to detect potholes</p>
        <input type="file" id="fileInput" accept="image/*" style="display: none;">
        <button onclick="document.getElementById('fileInput').click()">Choose Image</button>
    </div>

    <div class="loading" id="loading">
        <p>🔍 Detecting potholes...</p>
    </div>

    <div class="result" id="result" style="display: none;">
        <h3 id="resultText"></h3>
        <img id="resultImage" alt="Detection Result">
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const resultText = document.getElementById('resultText');
        const resultImage = document.getElementById('resultImage');
        const confSlider = document.getElementById('confidence');
        const confValue = document.getElementById('conf-value');

        confSlider.addEventListener('input', (e) => {
            confValue.textContent = e.target.value;
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processFile(e.target.files[0]);
            }
        });

        async function processFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            loading.style.display = 'block';
            result.style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);
            formData.append('confidence', confSlider.value);

            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    resultText.textContent = data.message;
                    resultImage.src = data.output_image;
                    result.style.display = 'block';
                } else {
                    alert('Detection failed');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
    """

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)