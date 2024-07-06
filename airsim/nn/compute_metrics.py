import torch

from models import SiameseDataset, SiameseNetwork, ContrastiveLoss, TrackToVector
from hparams import config


def calc_accuracy(y_pred, y):
    '''
    Расчет точности
    :param y_pred: предсказанный ответ
    :param y: реальный ответ
    :return: точность предсказаний
    '''
    return torch.mean(torch.tensor(y_pred == y, dtype=torch.float32))


def calc_f_beta(y_pred, y, beta=1.0):
    '''
    Расчет метрики F_beta
    :param y_pred: предсказанный ответ
    :param y: реальный ответ
    :param beta: параметр beta
    :return: значение метрики F_beta
    '''
    t = torch.cat((y_pred, y), 1)

    true_positive = t[(t[:, 0] == 0) & (t[:, 1] == 0), :].size(0)
    false_positive = t[(t[:, 0] == 0) & (t[:, 1] == 1), :].size(0)
    true_negative = t[(t[:, 0] == 1) & (t[:, 1] == 1), :].size(0)
    false_negative = t[(t[:, 0] == 1) & (t[:, 1] == 0), :].size(0)
    print(f'TP = {true_positive}; FP = {false_positive}; TN = {true_negative}; FN = {false_negative}')

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print(f'precision = {precision}; recall = {recall}')

    f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    print(f'F_{beta} = {f_beta}')

    return f_beta


def main():
    test_dataset = SiameseDataset(path='data/valid')
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset))

    model = TrackToVector()
    criterion = ContrastiveLoss(margin=config['loss_margin'], alpha=config['loss_alpha'])

    model.load_state_dict(torch.load('TrackToVector.pt'))

    for x_1, x_2, y in test_dataloader:
        with torch.inference_mode():
            out_1 = model(x_1)
            out_2 = model(x_2)

            loss = criterion(out_1, out_2, y)
            print(f'Loss = {loss.item()}')

            dist = torch.nn.functional.pairwise_distance(out_1, out_2, keepdim=True)
            y_pred = dist >= 0.1  # config['loss_margin'] / 2
            print(f'Accuracy = {calc_accuracy(y_pred, y)}')
            print(f'F_1 = {calc_f_beta(y_pred, y)}')

            X = torch.cat((y, dist), 1)
            print(list([d.item() for d in list(dist)]))
            print('Distance range between identical objects: [{d_min}, {d_max}]'.format(
                d_min=X[X[:, 0] == 0, :][:, 1].min(), d_max=X[X[:, 0] == 0, :][:, 1].max()
            ))
            print('Distance range between different objects: [{d_min}, {d_max}]'.format(
                d_min=X[X[:, 0] == 1, :][:, 1].min(), d_max=X[X[:, 0] == 1, :][:, 1].max()
            ))


if __name__ == '__main__':
    main()
