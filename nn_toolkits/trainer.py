import numpy as np

import torch as t
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm, trange
from IPython.display import clear_output

from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0, last_step=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineScheduler, self).__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif self.last_epoch < self.total_steps:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min for _ in self.base_lrs]


class Trainer:
    def __init__(self, model, train, criterion, batch_size=32, num_epochs=50, lr=3e-4,
                 eta_min=1e-6, warmup_steps=500, valid=None, max_norm=None):

        # Используем по возможности CUDA
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.model = model

        self.epochs = num_epochs
        self.batch_size = batch_size

        # Создаем загрузчики данных для оптимизации памяти
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
        self.valid_loader = None if valid is None else DataLoader(valid, batch_size=batch_size, num_workers=0)

        self.criterion = criterion

        # Если оптимизатора нет, тогда по дефолту ставим Adam
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineScheduler(self.optimizer, warmup_steps=warmup_steps,
                                         total_steps=len(self.train_loader) * num_epochs, eta_min=eta_min)

        self.max_norm = max_norm

        self.report = {  # Будем хранить отчеты в списке
            'train_losses': [],
            'valid_losses': [],
            'learning_rates': [self.scheduler.get_last_lr()]
        }

    @staticmethod
    def plot_losses(losses: list, n_part: int, label: str):
        """
        Отображает графики потерь.

        :param losses: Список всех потерь.
        :param n_part: До скольки элементов хотим уменьшить losses.
        :param label: Надпись в графике.
        """
        compacted_losses = [np.mean(losses[part * len(losses) // n_part: (part + 1) * len(losses) // n_part])
                            for part in range(n_part)]
        plt.plot(compacted_losses, label=label)

    def train(self, tqdm_disable=False):
        """
        Тренировка модели.

        :param tqdm_disable: Есть возможность выключить tqdm.
        """
        self.model.to(self.device)

        epochs_range = trange(self.epochs, leave=False) if not tqdm_disable else range(self.epochs)

        for _ in epochs_range:
            self.model.train()

            # Берем среднее только с последней эпохи
            train_desc = np.nan if len(self.report['train_losses']) == 0 \
                else np.mean(self.report['train_losses'][len(self.report['train_losses']) - len(self.train_loader):])
            train_iter = tqdm(self.train_loader, desc=f'Train Loss прошлой эпохи: {train_desc}', leave=False) \
                if not tqdm_disable else self.train_loader

            # Основное обучение
            for batch in train_iter:
                X_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                self.optimizer.zero_grad()

                logit = self.model(X_batch).squeeze()
                loss = self.criterion(logit, y_batch)

                loss.backward()

                # Gradient Clipping
                if self.max_norm:
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

                self.optimizer.step()

                self.report['train_losses'].append(loss.item())

                self.scheduler.step()
                self.report['learning_rates'].append(self.scheduler.get_last_lr())

            # При наличии валидационных данных оцениваем и их потери
            if self.valid_loader is not None:
                self.model.eval()
                with t.no_grad():

                    valid_desc = np.nan if len(self.report['valid_losses']) == 0 \
                        else np.mean(self.report['valid_losses'][len(self.report['valid_losses']) - len(self.valid_loader):])
                    valid_iter = tqdm(self.valid_loader, desc=f'Valid Loss прошлой эпохи: {valid_desc}', leave=False) \
                        if not tqdm_disable else self.valid_loader

                    for batch in valid_iter:
                        X_batch = batch[0].to(self.device)
                        y_batch = batch[1].to(self.device)

                        loss = self.criterion(self.model(X_batch).squeeze(), y_batch)
                        self.report['valid_losses'].append(loss.item())

        self.model.to('cpu')

        # Если tqdm не будет, то и стирать будет нечего
        if not tqdm_disable:
            # Удаляем tqdm
            clear_output()

    def visualize(self, averaging_rate=1):
        """
        Визуализирует ход обучения модели.

        :param averaging_rate: Bool or Int. Если bool, тогда возвращаем график потерь в каждой эпохе, а если int, то
        можем регулировать гладкость loss'а.
        """
        plt.figure(figsize=(16, 5))

        # Losses Graphics ##################
        plt.subplot(1, 2, 1)
        plt.title('Кривые потерь')

        self.plot_losses(self.report['train_losses'], n_part=int(self.epochs * averaging_rate),
                            label='Обучающая выборка')

        if len(self.report['valid_losses']) > 0:
            self.plot_losses(self.report['valid_losses'], n_part=int(self.epochs * averaging_rate),
                                label='Валидационная выборка')

        plt.grid(True)
        plt.legend()

        # Learning Rate Graphic ############
        plt.subplot(1, 2, 2)
        plt.title('Изменение learning rate')

        plt.plot(self.report['learning_rates'])

        plt.grid(True)
        plt.show()
