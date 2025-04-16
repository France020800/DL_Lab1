import os

from models.ResidualMLP import ResidualMLP
from comet_ml import start
from comet_ml.integration.pytorch import log_model

import utils
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    hyper_params = {
        'epochs': 25,
        'lr': 0.0001,
        'batch_size': 128,
        'input_size': 28 * 28,
        'output_size': 10,
        'width': 512,
        'depth': 10
    }

    model = ResidualMLP(hyper_params['input_size'], hyper_params['output_size'], hyper_params['width'], hyper_params['depth']).to(device)
    utils.start_train(model, hyper_params, comet_project='ResidualMLP MNIST', device=device)