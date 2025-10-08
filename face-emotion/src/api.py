from fastapi import FastAPI, File, UploadFile
import uvicorn
import io
from PIL import Image
import torch
from torchvision import transforms
from src.model import get_model, LABELS

app = FastAPI(title='Face Emotion API')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'weights/best_model.pth'

model = get_model(num_classes=len(LABELS), pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    return {'label': LABELS[top_idx], 'confidence': float(probs[top_idx]), 'probs': {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
