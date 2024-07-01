import time
import os
import torch as T
import json
from abc import abstractmethod, ABC
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict
from tqdm import tqdm
from collections import deque
from constants import *
from utils.logger import get_logger

@dataclass
class Tracker:
    last_loss: float
    last_metric_val: float
    epoch: int
    step_counter: int
    best_epoch: int
    best_metric_val: float

    def inc_step_counter(self):
        self.step_counter += 1

    def update_metric_val(self, metric_val: float, epoch: int):
        pass

class BaseTrainer(ABC):
    def __init__(self, model: T.nn.Module, gpu_id: int, args: Dict, log_enabled: bool = True, is_eval: bool = False):
        self.logger = get_logger(__class__.__name__, gpu_id)
        self.model = model
        self.args = args
        self.gpu_id = gpu_id
        self.log_enabled = log_enabled
        self.is_eval = is_eval

        if args.get('uid') is not None:
            self.uid = args['uid']
        else:
            self.uid = int(time.time())

        self.loss_fn = self._get_loss_fn()
        if not is_eval:
            self.optim = self._get_optimizer()
            self.scaler = T.cuda.amp.GradScaler()
            self.scheduler = self._get_scheduler()

        if log_enabled and self.gpu_id == 0:
            self.args['log_dir'] = os.path.join(args['log_dir'], f'{self.uid}')
            self.summary_writer = SummaryWriter(log_dir=self.args['log_dir'])
            self.args['ckpt_dir'] = os.path.join(self.args['log_dir'], 'weights')
            os.makedirs(self.args['ckpt_dir'], exist_ok=True)
            self.save_config()

        self.tracker = Tracker()
        # self.last_loss = None
        # self.last_metric_val = None
        # self.counter = 0
        # self.best_epoch = None
        # self.best_metric_val = None
        # self.history = {}
        self._ddp_model()        
    
    @abstractmethod
    def _get_loss_fn(self) -> T.nn.Module:
        raise NotImplementedError()
    
    @abstractmethod
    def _get_optimizer(self) -> T.optim.Optimizer:
        raise NotImplementedError()

    @abstractmethod
    def _get_scheduler(self) -> T.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()

    @abstractmethod
    def step(self, *batch_data, is_train: bool):
        raise NotImplementedError()

    def _ddp_model(self):
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    @property
    def is_main_process(self):
        return self.gpu_id == 0

    @property
    def can_log(self):
        return self.log_enabled and self.is_main_process

    def is_metric_val_better(self, epoch=None):
        if self.best_metric_val is None or self.last_metric_val > self.best_metric_val:
            self.best_metric_val = self.last_metric_val
            self.best_epoch = epoch
            return True
        return False

    def write_summary(self, title: str, value: float, step: int):
        if self.can_log:
            self.summary_writer.add_scalar(title, value, step)
    
    def write_image(self, title: str, image: float, step: int):
        if self.can_log:
            self.summary_writer.add_image(title, image, step)

    def save_config(self):
        if self.is_main_process:
            config = self.args

            self.logger.info('======CONFIGURATIONS======')
            for k, v in config.items():
                self.logger.info(f'{k.upper()}: {v}')
            
            config_path = os.path.join(self.args['log_dir'], 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)
            self.logger.info(f'Training config saved to {config_path}')

    def save_checkpoint(self, epoch: int, name: str = '', only_model: bool = True):
        if self.is_main_process:
            save_checkpoint = {'model': self.model.module.state_dict()}
            if not only_model:
                save_checkpoint['optimizer'] = self.optim.state_dict()
                save_checkpoint['scheduler'] = self.scheduler.state_dict()
            if name != '':
                ckpt_path = os.path.join(self.args['ckpt_dir'], f'{name}.pt')
            else:
                ckpt_path = os.path.join(
                    self.args['ckpt_dir'],
                    f'epoch{epoch:02}_loss{self.last_loss:.4f}_metric{self.last_metric_val:.4f}.pt',
                )
            T.save(save_checkpoint, ckpt_path)
            self.logger.info(f'Checkpoint saved to {ckpt_path}')
    
    def load_checkpoint(self, ckpt_path: str, only_model: bool = True):
        assert os.path.exists(ckpt_path)
        checkpoint = T.load(ckpt_path, map_location='cpu')
        self.model.module.load_state_dict(checkpoint['model'])
        if not only_model:
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.logger.info(f'Succesfully loaded model in {ckpt_path}')

    def process_data(self, dl: T.utils.data.DataLoader, is_train: bool, epoch: int):
        if is_train:
            self.logger.info('Training Phase')
        elif not self.is_eval:
            self.logger.info('Validation Phase')

        batch_losses = T.zeros(2, device=self.gpu_id)

        pbar = tqdm(dl, disable=not self.is_main_process)

        for i, batch_data in enumerate(pbar):
            if not is_train:
                self.model.eval()
                with T.no_grad():
                    b_loss = self.step(batch_data, is_train)
            else:
                self.model.train()
                b_loss = self.step(batch_data, is_train)
                self.scheduler.step()

                for i in range(len(self.optim.param_groups)):
                    self.write_summary(f'LR Scheduler/{i}', self.optim.param_groups[i]['lr'], self.counter)
                self.write_summary('Training/Batch Loss', b_loss, self.counter)

                self.counter += 1
                yield i

            if not self.is_main_process:  # reset for gpu rank > 0
                batch_losses = T.zeros(2, device=self.gpu_id)

            batch_losses[0] += b_loss
            batch_losses[1] += 1

            T.distributed.reduce(batch_losses, dst=0)
            avg_losses = batch_losses[0] / batch_losses[1]
            
            pbar.set_postfix({'Loss': f'{avg_losses:.5f}'})

        if not is_train:
            self.last_loss = avg_losses
            tag = 'Validation'
        else:
            tag = 'Training'

        self.write_summary(f'{tag}/Loss', avg_losses, epoch)
        yield -1

    
    def do_training(self, train_dataloader: T.utils.data.DataLoader, val_dataloader: T.utils.data.DataLoader):
        eval_per_epoch = self.args['eval_per_epoch']
        epoch = self.args['epoch']
        
        eval_idx = [len(train_dataloader) // eval_per_epoch * i for i in range(1, eval_per_epoch)]
        early_stop = False
        for i in range(epoch):
            self.logger.info(f'Epoch {i+1}/{epoch}')
            k = 0
            for step in self.process_data(train_dataloader, True, i):
                if step in eval_idx or step == -1:
                    deque(self.process_data(val_dataloader, False, eval_per_epoch * i + k), maxlen=0)

                    if self.is_metric_val_better(i + 1):
                        self.save_checkpoint(i + 1, 'best')
                    else:
                        if self.args['patience'] > 0 and i + 1 - self.best_epoch > self.args['patience']:
                            early_stop = True
                            break
                    k += 1

            if (i + 1) % self.args['ckpt_interval'] == 0 or i == self.args['epoch'] - 1:
                self.save_checkpoint(i + 1)

            self.logger.info(f'Epoch complete\n')

            if early_stop:
                self.logger.info(f'Early stopping. No improvement in validation metric for the last {self.args["patience"]} epochs.')
                break
        self.logger.info(f'Best result was seen in epoch {self.best_epoch}')
        
        final_result = {
            'best_epoch': self.best_epoch,
            'best_metric_val': self.best_metric_val,
        }
        final_result_path = os.path.join(self.args['log_dir'], 'final_result.json')
        with open(final_result_path, 'w') as f:
            json.dump(final_result, f)
        self.logger.info(f'Final result saved to {final_result_path}')

    def do_evaluation(self, test_dataloader: T.utils.data.DataLoader):
        deque(self.process_data(test_dataloader, False, 0), maxlen=0)
        self.logger.info(f'Loss: {self.last_loss:.5f}')
