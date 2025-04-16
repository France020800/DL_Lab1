from models.CNN import CNN
from models.ResidualCNN import ResidualCNN

import utils
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    hyper_params = {
        'epochs': 25,
        'lr': 0.0001,
        'batch_size': 128,
        'in_channels': 1,
        'conv_channels': [512, 256, 128, 64, 32, 16],
    }

    model = ResidualCNN(in_channels=hyper_params['in_channels'], conv_channels=hyper_params['conv_channels'], num_classes=10).to(device)
    #model = CNN(in_channels=hyper_params['in_channels'], conv_channels=hyper_params['conv_channels'], num_classes=10).to(device)
    utils.start_train(model, hyper_params, comet_project='Residual CNN MNIST', device=device)