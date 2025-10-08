import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.dataset import FERDataset, get_transforms    # note: use package import in VS Code if you add project root to PYTHONPATH
from src.model import get_model, LABELS

def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Device:', device)

    dataset = FERDataset(csv_path=args.csv, transform=get_transforms(train=True, size=args.size))
    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(num_classes=args.num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for imgs, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{args.epochs}'):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(outputs, labels) * imgs.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_acc / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(outputs, labels) * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            print('Saved best model.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/raw/fer2013.csv')
    parser.add_argument('--out_dir', type=str, default='weights')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    args = parser.parse_args()
    train(args)
