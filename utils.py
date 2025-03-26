import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def train_epoch(model, train_loader, device, optimizer, loss_function = F.cross_entropy, current_epoch='unknown'):
    model.train()
    current_losses = []
    for (x, Y) in tqdm(train_loader, desc=f'Start training epoch {current_epoch}'):
        optimizer.zero_grad()
        x = x.to(device)
        Y = Y.to(device)
        out = model(x)
        current_loss = loss_function(out, Y)
        current_loss.backward()
        optimizer.step()
        current_losses.append(current_loss.item())
    return np.mean(current_losses)


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    ground_truths = []
    for (x, Y) in tqdm(test_loader, desc='Evaluating', leave=False):
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)
        ground_truths.append(Y)
        predictions.append(preds.detach().cpu().numpy())

    return (accuracy_score(np.hstack(ground_truths), np.hstack(predictions)),
            classification_report(np.hstack(ground_truths), np.hstack(predictions), zero_division=0, digits=3))

def plot_validation_curves(losses, accs):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(accs, color='deepskyblue')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Best Accuracy = {np.max(accs)} @ epoch {np.argmax(accs)}')