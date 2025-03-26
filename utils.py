import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
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


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Sottraggo la media e divido per la deviazione standard
        ])

        ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
        ds_test = MNIST(root='./data', train=False, download=True, transform=transform)

        val_size = 5000
        I = np.random.permutation(len(ds_train))
        ds_val = Subset(ds_train, I[:val_size])
        ds_train = Subset(ds_train, I[val_size:])

        train_loader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=4)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')

    return train_loader, validation_loader, test_loader

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