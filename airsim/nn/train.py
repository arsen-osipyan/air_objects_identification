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
    epoch_accuracy_history = []
    loss_history = []
    accuracy_history = []
    lr_history = []

    epoch_progress = trange(config['n_epochs'])
    for epoch in epoch_progress:
        epoch_progress.set_description(f'Epoch {epoch + 1}')

        batch_progress = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x_1, x_2, y) in batch_progress:
            # out_1, out_2 = model(x_1, x_2)
            out_1 = model(x_1)
            out_2 = model(x_2)

            loss = criterion(out_1, out_2, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dist = torch.nn.functional.pairwise_distance(out_1, out_2, keepdim=True)
            y_pred = (dist >= config['loss_margin'] / 2).float()
            diff = 1.0 - torch.abs(y - y_pred)

            epoch_loss_history.append(loss.item())
            epoch_accuracy_history.append(torch.mean(diff).item())

            batch_progress.set_postfix({'loss': loss.item(), 'accuracy': torch.mean(diff).item()})

        lr_history.append(optimizer.param_groups[0]['lr'])
        loss_history.append(np.mean(epoch_loss_history))
        accuracy_history.append(np.mean(epoch_accuracy_history))
        epoch_loss_history.clear()
        epoch_accuracy_history.clear()

        if scheduler is not None:
            scheduler.step()

        epoch_progress.set_postfix({'epoch_mean_loss': loss_history[-1], 'epoch_mean_accuracy': accuracy_history[-1]})
        print(f'Epoch {epoch + 1} -- Loss: {loss_history[-1]} Accuracy: {accuracy_history[-1]}')

    torch.save(model.state_dict(), 'TrackToVector.pt')

    return loss_history, lr_history


def save_plots(loss_history, lr_history):
    fig, axis = plt.subplots(1, 2, figsize=(12, 6))

    axis[0].set_title('Epoch Mean Loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].semilogy()
    axis[0].grid()
    axis[0].plot(loss_history)

    axis[1].set_title('Learning Rate')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Learning Rate')
    axis[1].grid()
    axis[1].plot(lr_history)

    plt.savefig('plots.png')


def main():
    train_dataset = SiameseDataset(path='data/train')
    # model = SiameseNetwork()
    model = TrackToVector()
    criterion = ContrastiveLoss(margin=config['loss_margin'], alpha=config['loss_alpha'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: 1.0)

    loss_history, lr_history = train(train_dataset, model, criterion, optimizer)

    save_plots(loss_history, lr_history)


if __name__ == '__main__':
    main()
