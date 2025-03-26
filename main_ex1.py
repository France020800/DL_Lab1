from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from Lab1.MLP import MLP

import utils
import numpy as np
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    epochs = 25
    lr = 0.0001
    batch_size = 128

    input_size = 28 * 28
    width = 32
    depth = 3

    model = MLP([input_size] + [width] * depth + [10]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_loader, validation_loader, test_loader = utils.load_dataset('mnist', batch_size=batch_size)

    losses = []
    accs = []
    writer = SummaryWriter()
    for epoch in range(epochs):
        loss = utils.train_epoch(model, train_loader, device, optimizer, current_epoch=epoch)
        (val_acc, _) = utils.evaluate_model(model, validation_loader, device=device)
        losses.append(loss)
        accs.append(val_acc)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
    writer.flush()
    writer.close()

    # And finally plot the curves.
    utils.plot_validation_curves(losses, accs)
    print(f'Accuracy report on TEST:\n {utils.evaluate_model(model, test_loader, device=device)[1]}')