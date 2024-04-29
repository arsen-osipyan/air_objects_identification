import torch


class SiameseDataset(torch.utils.data.Dataset):

    def __init__(self, path, x_file='x.pt', y_file='y.pt'):
        self.x = torch.load(path + '/' + x_file)
        self.y = torch.load(path + '/' + y_file)

        if self.x.shape[0] != self.y.shape[0]:
            raise RuntimeError(f'Tensors X and Y must have the same number of items.')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx][0], self.x[idx][1], self.y[idx]


class SiameseNetwork(torch.nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=16, out_channels=64, kernel_size=9),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=256, kernel_size=9),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8),
            torch.nn.LeakyReLU(),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 2),
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def forward(self, x_1, x_2):
        out_1 = self.forward_once(x_1)
        out_2 = self.forward_once(x_2)

        return out_1, out_2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0, alpha=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, out_1, out_2, y):
        euclidean_distance = torch.nn.functional.pairwise_distance(out_1, out_2, keepdim=True)

        loss_contrastive = torch.mean((1 - y) * torch.pow(euclidean_distance, self.alpha) +
                                      (y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                      self.alpha))

        return loss_contrastive