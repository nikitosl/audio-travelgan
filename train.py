from data import AudioData
from trainer import TravelGAN
from torch.utils.data.dataloader import DataLoader
from utils import get_device, load_json, get_writer
from statistics import mean
import numpy as np


def train_model(row_dataset_a, row_dataset_b, hparams, log, device=0):

    device = get_device(device)
    print('Loading data..')
    L = hparams["L"]

    dataset_a = AudioData(row_dataset_a)
    dataset_b = AudioData(row_dataset_b)

    loader_a = DataLoader(dataset_a, **hparams['loading'])
    loader_b = DataLoader(dataset_b, **hparams['loading'])
    print(f'Shape of dsA: {dataset_a.data.shape}')
    print(f'Shape of dsB: {dataset_b.data.shape}')

    model = TravelGAN(**hparams['model'], L=L, device=device)
    model.double()
    writer, monitor = get_writer(log)
    print('Start training..')

    for epoch in range(hparams['n_epochs']):
        # Run one epoch
        dis_losses, gen_losses = [], []
        for x_a, x_b in zip(loader_a, loader_b):
            # Loading on device
            x_a = x_a.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)

            # Calculate losses and update weights
            dis_loss = model.dis_update(x_a, x_b)
            gen_loss = model.gen_update(x_a, x_b)
            dis_losses.append(dis_loss)
            gen_losses.append(gen_loss)

        # Logging losses
        dis_loss, gen_loss = mean(dis_losses), mean(gen_losses)
        writer.add_scalar('dis', dis_loss, epoch)
        writer.add_scalar('gen', gen_loss, epoch)
        print(monitor.format(epoch, gen_loss, dis_loss))

        # Saving model every n_save_steps epochs
        if (epoch + 1) % hparams['n_save_steps'] == 0:
            model.save(log, epoch)

    return model


if __name__ == '__main__':

    dsA = np.load(r'./samples/mellog_shaped_digits.npy')
    dsB = np.load(r'./samples/mellog_shaped_men.npy')

    train_model(dsA, dsB,
                hparams=load_json('./configs', 'audata_conf'),
                log='logging')


