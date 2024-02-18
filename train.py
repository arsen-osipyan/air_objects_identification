import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm, trange

from hparams import config


class TwoLayersNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayersNN, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, input_size).double()
        self.fc2 = torch.nn.Linear(input_size, output_size).double()

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def print(self):
        print(self.fc1)


def train(model, optimizer, criterion, x_train, y_train, n_epochs=20000, target_loss=0.25, enable_plots=True,
          message_rate=1):
    losses = []
    lrs = []
    progressbar = tqdm(range(n_epochs))
    for epoch in progressbar:
        if epoch % message_rate == (message_rate - 1):
            progressbar.set_description(f'Epoch {epoch + 1}')

        output = model(x_train)
        loss = criterion(output, y_train)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 10 ** 10)
        optimizer.step()
        optimizer.zero_grad()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        if epoch % message_rate == (message_rate - 1):
            progressbar.set_postfix({'loss': loss.item()})

        # scheduler.step()
        # data_fold_width = 100/
        # if epoch % data_fold_width == data_fold_width - 1 and np.std(losses[-data_fold_width:]) / np.mean(losses[-data_fold_width:]) < 0.025:
        # print(optimizer.param_groups[0]['lr'], end=' -> ')
        # optimizer.param_groups[0]['lr'] *= 0.5
        # print(optimizer.param_groups[0]['lr'])

        if loss.item() < target_loss:
            break

    if enable_plots:
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))

        axis[0].set_title("Loss");
        axis[1].set_title("Learning rate")
        axis[0].set_xlabel("N_epoch");
        axis[1].set_xlabel("N_epoch")
        axis[0].set_ylabel("Loss");
        axis[1].set_ylabel("Learning rate")
        axis[0].semilogy();  # axis[1].semilogy()
        axis[0].grid();
        axis[1].grid()
        axis[0].plot(losses, label='train');
        axis[1].plot(lrs)
        axis[0].legend();


n_epochs = 7000
model = Simple2LayersOptimizer(input_size=x_train.shape[1], output_size=y_train.shape[1])
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
criterion = nn.MSELoss()
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

train(model, optimizer, criterion, x_train, y_train, n_epochs=n_epochs, target_loss=0.001)


def main():
    train_dataset = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=True)

    model = TwoLayersNN(input_size=20, output_size=1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])

    lrs = []
    losses = []
    for epoch in trange(config["n_epochs"]):
        for i, (features, target) in enumerate(tqdm(train_loader)):
            output = model(features)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())

    torch.save(model.state_dict(), "model.pt")

    fig, axis = plt.subplots(1, 2, figsize=(12, 6))

    axis[0].set_title("Loss")
    axis[0].set_xlabel("N_epoch")
    axis[0].set_ylabel("Loss")
    axis[0].semilogy()
    axis[0].grid()
    axis[0].plot(losses)

    axis[1].set_title("Learning rate")
    axis[1].set_xlabel("N_epoch")
    axis[1].set_ylabel("Learning rate")
    axis[1].grid()
    axis[1].plot(lrs)

    fig.savefig('stats.png')


if __name__ == "__main__":
    main()
