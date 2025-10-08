import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

def plot_losses(history):
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.legend()
    plt.show()

def get_class_report(model, dataloader, device):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    print(classification_report(targets, preds))
    print(confusion_matrix(targets, preds))
