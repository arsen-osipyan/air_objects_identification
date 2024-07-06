import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange

from models import SiameseDataset, SiameseNetwork, ContrastiveLoss, TrackToVector
from hparams import config


def train(dataset, model, criterion, optimizer, scheduler=None):
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             shuffle=config['shuffle'],
                                             batch_size=config['batch_size'])

    epoch_loss_history = []
    loss_history = []

    for epoch in range(config['n_epochs']):

        batch_progress = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
        for i, (x_1, x_2, y) in batch_progress:
            batch_progress.set_description(f'Epoch {epoch + 1}')

            out_1 = model(x_1)
            out_2 = model(x_2)

            loss = criterion(out_1, out_2, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_history.append(loss.item())

            batch_progress.set_postfix({'loss': loss.item()})

        loss_history.append(np.mean(epoch_loss_history))
        epoch_loss_history.clear()

        if scheduler is not None:
            scheduler.step()

        print(f'Epoch {epoch + 1} -- Loss: {loss_history[-1]}')

    torch.save(model.state_dict(), 'TrackToVector.pt')

    return loss_history


def plot_loss(loss_history):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title('Средние значения функции потерь', fontsize=22)
    ax.set_xlabel('Эпоха', fontsize=16)
    ax.set_ylabel('Значение функции потерь', fontsize=16)

    ax.plot(loss_history)

    ax.grid()

    plt.savefig('loss.png')


def main():
    train_dataset = SiameseDataset(path='data/train')
    model = TrackToVector()
    criterion = ContrastiveLoss(margin=config['loss_margin'], alpha=config['loss_alpha'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    loss_history = train(train_dataset, model, criterion, optimizer)

    plot_loss(loss_history)


if __name__ == '__main__':
    main()
