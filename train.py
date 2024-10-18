import os
import argparse
import yaml
import random
import torch as T
import numpy as np
import torch.multiprocessing as mp
from typing import Dict
from utils.logger import *
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from data.example_data import ExampleData
from model.example_model import ExampleModel
from trainer.example_trainer import ExampleTrainer

def ddp_setup(rank: int, world_size: int, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)

def main(rank: int, world_size: int, train_args: Dict, port: int):
    setup_logging()
    seed_everything(train_args['train']['seed'])
    if not train_args['train']['no_ddp']:
        ddp_setup(rank, world_size, port)
    
    logger = get_logger(__name__, rank)

    logger.info('Instantiating model and trainer agent')
    model = ExampleModel(**train_args['model'])
    trainer = ExampleTrainer(model, rank, train_args, log_enabled=not train_args['train']['no_save'])

    logger.info('Preparing dataset')
    train_dataset = ExampleData(**train_args['data'], validation=False)
    val_dataset = ExampleData(**train_args['data'], validation=True)
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')

    logger.info(f'Using {world_size} GPU(s), actual batch size: {train_args["train"]["batch_size"] * world_size}')
    if train_args['train'].get('model_path') is not None:
        trainer.load_checkpoint(train_args['model_path'])

    logger.info('Instantiating dataloader')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_args['train']['batch_size'],
        shuffle=True if train_args['train']['no_ddp'] else False,
        num_workers=train_args['train']['n_workers'],
        pin_memory=True,
        sampler=None if train_args['train']['no_ddp'] else DistributedSampler(train_dataset),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_args['train']['batch_size'],
        shuffle=False,
        num_workers=train_args['train']['n_workers'],
        pin_memory=True,
        sampler=None if train_args['train']['no_ddp'] else DistributedSampler(val_dataset),
    )

    trainer.do_training(train_dataloader, val_dataloader)

    if not train_args['train']['no_ddp']:
        destroy_process_group()

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('QuickTorch Train', add_help=False)
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--config', type=str, help='path to json config', default='config/example_config.yaml')
    parser.add_argument('--model-path', type=str, help='ckpt path to continue', default=None)
    parser.add_argument('--patience', type=int, help='patience for early stopping', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--port', type=int, help='DDP port', default=None)
    parser.add_argument('--no-ddp', action='store_true', help='disable DDP')
    parser.add_argument('--no-save', action='store_true', help='disable logging and checkpoint saving (for debugging)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    train_args = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.seed is None and train_args['train'].get('seed', None) is None:
        train_args['train']['seed'] = random.randint(0, 1000000)
    if args.uid is not None:
        train_args['train']['uid'] = args.uid
    if args.model_path is not None:
        train_args['train']['model_path'] = args.model_path
    if args.patience != -1:
        train_args['train']['patience'] = args.patience 
    train_args['train']['no_ddp'] = args.no_ddp
    train_args['train']['no_save'] = args.no_save

    if not train_args['train']['no_ddp']:
        world_size = T.cuda.device_count()
        port = str(random.randint(10000, 60000)) if args.port is None else str(args.port)
        mp.spawn(main, nprocs=world_size, args=(world_size, train_args, port))
    else:
        main(0, 1, train_args, 0)