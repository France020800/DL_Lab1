import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
from comet_ml import start
from comet_ml.integration.pytorch import log_model
from sklearn.metrics import accuracy_score, classification_report


def start_train(model, hyper_params, comet_project, dataset='mnist', device='cpu'):
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyper_params['lr'])

    train_loader, validation_loader, test_loader = load_dataset('mnist', batch_size=hyper_params['batch_size'])

    losses = []
    accs = []

    my_var = os.getenv("comet_api_key")
    experiment = start(
        api_key=my_var,
        project_name=comet_project,
        workspace="france020800",
    )
    experiment.add_tag("pytorch")
    experiment.log_parameters(hyper_params)

    experiment.set_model_graph(str(model))

    for epoch in range(hyper_params['epochs']):
        experiment.log_current_epoch(epoch)
        loss = train_epoch(model, train_loader, device, optimizer, current_epoch=epoch)
        (val_acc, _) = evaluate_model(model, validation_loader, device=device)
        losses.append(loss)
        accs.append(val_acc)
        experiment.log_metrics({
            'loss': loss,
            'val_acc': val_acc
        })

    # Log the model to Comet for easy tracking and deployment
    log_model(experiment, model, "MLP-MNIST")

    # Capture the accuracy report
    accuracy_report = evaluate_model(model, test_loader, device=device)[1]

    # Log the accuracy report to Comet
    experiment.log_text("Accuracy report on TEST:\n" + accuracy_report)

    # And finally plot the curves.
    plot_validation_curves(losses, accs)
    print(f'Accuracy report on TEST:\n {accuracy_report}')

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
    elif dataset_name == 'cifar-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Sottraggo la media e divido per la deviazione standard
        ])

        ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
        ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

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