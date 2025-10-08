# Face Emotion Detection - Quick README

## Setup
1. Create & activate venv (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate

2. Install
   pip install -r requirements.txt

3. Train (example)
   python .\src\train.py --csv data/raw/fer2013.csv --out_dir weights --epochs 12 --batch 64

4. Inference (webcam)
   python .\src\infer.py

5. API
   python .\src\api.py
