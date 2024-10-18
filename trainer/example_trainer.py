import time
import os
import torch as T
import yaml
from typing import Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from constants import *
from utils.logger import get_logger
from trainer.base_trainer import BaseTrainer, Tracker
from torch_cosine_annealing import CosineAnnealingWithWarmRestarts

class ExampleTrainer(BaseTrainer):
    def __init__(self, model: T.nn.Module, gpu_id: int, args: Dict, log_enabled: bool = True, is_eval: bool = False):
        self.logger = get_logger(__class__.__name__, gpu_id)
        super().__init__(model, gpu_id, args, log_enabled, is_eval)
        self.tracker = Tracker(direction='min')
    
    def _get_optimizer(self) -> T.optim.Optimizer:
        return T.optim.AdamW(lr=self.args['train']['lr'], params=self.model.parameters(), betas=(0.9, 0.999), eps=1e-15)

    def _get_scheduler(self) -> T.optim.lr_scheduler.LRScheduler:
        return CosineAnnealingWithWarmRestarts(
            optimizer=self.optim,
            cycle_period=self.args['train']['cycle_period'],
            cycle_mult=self.args['train']['cycle_mult'],
            min_lr=self.args['train']['min_lr'],
            gamma=self.args['train']['lr_decay'],
            strategy='step',
        )
    
    def _get_loss_fn(self) -> T.nn.Module:
        return T.nn.MSELoss()

    def step(self, *batch_data) -> T.Tensor:
        x, y = batch_data
        batch_data = [x.to(self.gpu_id) for x in batch_data if type(x) == T.Tensor]

        x, y = batch_data

        with T.cuda.amp.autocast():
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

        return loss
