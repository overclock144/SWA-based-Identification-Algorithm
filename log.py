import os
import time

from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter


class Log:
    def __init__(self, batch_size, epochs, dataset, dataset_name, writer_dir='runs', time_func=time.time, format=True):
        """
        实例化后调用start_log()

        epoch循环内dataloader前调用epoch_start()

        epoch循环内dataloader内调用iteration()

        epoch循环内dataloader内调用
        calculate_acc() # 必选(正确调用iteration才可以输出结果)
        calculate_loss() # 必选(正确调用iteration才可以输出结果)
        output_writer() # 可选

        epoch循环内dataloader后调用epoch_end()

        训练结束后调用end_log()


        :param dataset_name: 数据集名称
        :param writer_dir: SummaryWriter日志目录
        :param time_func: 时间函数,例如time.time
        :param format: True为进度条式风格,False为可保存文档风格
        """
        # 实例化SummaryWriter
        self.writer = SummaryWriter(writer_dir)
        # 时间记录数组
        self.list_time = []
        # 时间记录器
        self.time_func = time_func
        # epoch记录
        self.epoch = 0
        self.epochs = epochs
        # epoch数据记录
        self.acc_total = []
        self.loss_total = []
        self.batch_step = []  # epoch batch step
        self.dataset_name = dataset_name
        # 其他
        self.format = format
        self.dataset = dataset
        self.batch_size = batch_size

        if not self.format:
            self.table = PrettyTable(['Dataset', 'Epoch', 'Iteration', 'Loss', 'Acc', 'Time'])

    def __call__(self, *args, **kwargs):
        pass

    def __del__(self):
        pass

    def epoch_start(self):
        self.list_time.append([self.time_func()])
        self.acc_total.append([])
        self.loss_total.append([])
        self.batch_step.append([0])
        if self.format:
            print('\r|', f'Epoch [{self.epoch}/{self.epochs}] {self.dataset_name} Start'.center(100, '-'), '|', end='')

    def iteration(self, n, i):
        if i % n == 0 and i != 0:  # n个iteration后
            self.list_time[self.epoch].append(self.time_func())
            if self.format:
                print('\r|', f'Dataset [{self.dataset_name}] '
                             f'| Epoch [{self.epoch}/{self.epochs}] '
                             f'| Loss [{sum(self.loss_total[self.epoch]) / (self.batch_size * (i + 1)) :.4f}] '
                             f'| Acc [{sum(self.acc_total[self.epoch]) / (self.batch_size * (i + 1)) :.4f}] '
                             f'| Time [{self.list_time[self.epoch][-1] - self.list_time[self.epoch][-2]:.2f}s] '
                             f'| Iteration [{(self.batch_step[self.epoch][0] + 1) * 10}/{round(len(self.dataset[self.dataset_name]) / self.batch_size)}]'.center(
                    100), '|', end='')
            else:
                self.table.add_row([f'{self.dataset_name}', f'{self.epoch}/{self.epochs}',
                                    f'{(self.batch_step[self.epoch][0] + 1) * 10}/{round(len(self.dataset[self.dataset_name]) / self.batch_size)}',
                                    f'{sum(self.loss_total[self.epoch]) / (self.batch_size * (i + 1)) :.4f}',
                                    f'{sum(self.acc_total[self.epoch]) / (self.batch_size * (i + 1)) :.4f}',
                                    f'{self.list_time[self.epoch][-1] - self.list_time[self.epoch][-2]:.2f}s'])
                os.system('cls')
                print(self.table)
            self.batch_step[self.epoch][0] = self.batch_step[self.epoch][0] + 1

    def epoch_end(self):
        if self.format:
            print('\r|', f'Epoch [{self.epoch}/{self.epochs}] {self.dataset_name} Completed'.center(100, '-'), '|')
            print('|', f'Dataset [{self.dataset_name}] '
                       f'| Epoch [{self.epoch}/{self.epochs}] '
                       f'| Loss [{sum(self.loss_total[self.epoch]) / len(self.dataset[f"{self.dataset_name}"]):.4f}] '
                       f'| Acc [{sum(self.acc_total[self.epoch]) / len(self.dataset[f"{self.dataset_name}"]):.4f}] '
                       f'| Time [{self.time_func() - self.list_time[self.epoch][0]:.2f}s]'.center(100), '|', end='')
        self.epoch += 1

    def get_step(self):
        return self.batch_step

    def calculate_loss(self, i_loss):
        """
        输入mini-batch的loss
        :param i_loss:
        :return:
        """
        self.loss_total[self.epoch].append(i_loss)

    def calculate_acc(self, i_acc):
        """
        输入mini-batch的acc(正确数量)
        :param i_acc:
        :return:
        """
        self.acc_total[self.epoch].append(i_acc)

    def output_writer(self, loss, acc):
        self.writer.add_scalar(f'{self.dataset_name} Loss {self.epoch}', loss.item() / self.batch_size,
                               self.batch_step[self.epoch][0])
        self.writer.add_scalar(f'{self.dataset_name} Acc {self.epoch}', acc.item() / self.batch_size,
                               self.batch_step[self.epoch][0])

    def start_log(self):
        # for i in range(101):
        #     print(f'\rLoading Data:{i * "▋"}', f'{i}%', end='')
        # print('\n')
        if self.format:
            print('┌', '-'.center(100, '-'), '┐')

    def end_log(self):
        if self.format:
            print('└', '-'.center(100, '-'), '┘')
