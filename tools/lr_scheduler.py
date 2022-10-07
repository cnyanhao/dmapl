import math
from typing import Optional
from torch.optim.optimizer import Optimizer


class StepwiseLR1:
    """
    lr * (1 + gamma * i / n) ** ( - decay)
    """
    def __init__(self, optimizer: Optimizer, max_iter: int, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 10, decay_rate: Optional[float] = 0.75):
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num / self.max_iter) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class StepwiseLR2:
    """
    lr * (1 + gamma * i) ** ( - decay)
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class StepwiseLR3:
    """
    fix lr
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01):
        self.init_lr = init_lr
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class StepwiseLR4:
    """
    cosine
    """
    def __init__(self, optimizer: Optimizer, max_iter: int, init_lr: Optional[float] = 0.01, end_lr: Optional[float] = None):
        self.max_iter = max_iter
        self.init_lr = init_lr
        if end_lr is not None:
            self.end_lr = end_lr
        else:
            self.end_lr = init_lr / 10
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.end_lr + 0.5 * (self.init_lr - self.end_lr) * (1 + math.cos(self.iter_num / self.max_iter * math.pi))
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1