import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import get_model, LABELS

def load_model(weights_path='weights/best_model.pth', device='cpu'):
    device = torch.device(device)
    model = get_model(num_classes=len(LABELS), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, device

def preprocess_face(face_bgr, size=224, device='cpu'):
    img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

def main(weights='weights/best_model.pth', cam_index=0, size=224, device_choice='cuda' if torch.cuda.is_available() else 'cpu'):
    model, device = load_model(weights_path=weights, device=device_choice)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print('Cannot open camera')
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            inp = preprocess_face(face, size=size, device=device)
            with torch.no_grad():
                out = model(inp)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = LABELS[idx]
            conf = probs[idx]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f\"{label} {conf:.2f}\", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('Emotion (q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
