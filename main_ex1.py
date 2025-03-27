import os

from comet_ml.config.config_api import experiment
from numpy import random

from MLP import MLP
from comet_ml import ConfusionMatrix, login, start
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
        'width': 128,
        'depth': 5
    }

    model = MLP([hyper_params['input_size']] + [hyper_params['width']] * hyper_params['depth'] + [10]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyper_params['lr'])

    train_loader, validation_loader, test_loader = utils.load_dataset('mnist', batch_size=hyper_params['batch_size'])

    losses = []
    accs = []

    my_var = os.getenv("comet_api_key")
    experiment = start(
        api_key=my_var,
        project_name="MLP MNIST",
        workspace="france020800",
    )
    experiment.add_tag("pytorch")
    experiment.log_parameters(hyper_params)

    experiment.set_model_graph(str(model))

    for epoch in range(hyper_params['epochs']):
        experiment.log_current_epoch(epoch)
        loss = utils.train_epoch(model, train_loader, device, optimizer, current_epoch=epoch)
        (val_acc, _) = utils.evaluate_model(model, validation_loader, device=device)
        losses.append(loss)
        accs.append(val_acc)
        experiment.log_metrics({
            'loss': loss,
            'val_acc': val_acc
        })

    # Log the model to Comet for easy tracking and deployment
    log_model(experiment, model, "MLP-MNIST")

    # Capture the accuracy report
    accuracy_report = utils.evaluate_model(model, test_loader, device=device)[1]

    # Log the accuracy report to Comet
    experiment.log_text("Accuracy report on TEST:\n" + accuracy_report)

    # And finally plot the curves.
    utils.plot_validation_curves(losses, accs)
    print(f'Accuracy report on TEST:\n {accuracy_report}')