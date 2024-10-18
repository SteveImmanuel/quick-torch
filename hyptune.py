import optuna
import os
import argparse
import time
import yaml
import random
import torch as T
import torch.multiprocessing as mp
from optuna.trial import BaseTrial
from train import main

def wrapper_objective(args):
    def objective(trial: BaseTrial):
        train_args = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        if args.seed is None and train_args['train'].get('seed', None) is None:
            train_args['train']['seed'] = random.randint(0, 1000000)
        if args.model_path is not None:
            train_args['train']['model_path'] = args.model_path
        if args.patience != -1:
            train_args['train']['patience'] = args.patience 
        train_args['train']['no_ddp'] = args.no_ddp
        train_args['train']['no_save'] = args.no_save
        
        train_args['train']['uid'] = int(time.time())
        train_args['train']['log_dir'] = 'logs/hyptune'
        
        train_args['model']['n_neurons'] = trial.suggest_categorical('n_neurons', [10, 20, 30])

        if not train_args['train']['no_ddp']:
            world_size = T.cuda.device_count()
            port = str(random.randint(10000, 60000)) if args.port is None else str(args.port)
            mp.spawn(main, nprocs=world_size, args=(world_size, train_args, port))
        else:
            main(0, 1, train_args, 0)

        trial.set_user_attr('uid', train_args['train']['uid'])
        result = yaml.load(open(f'{train_args["train"]["log_dir"]}/{train_args["train"]["uid"]}/result.yaml', 'r'), Loader=yaml.FullLoader)
        return result['best_metric']
    return objective

def get_args():
    parser = argparse.ArgumentParser('QuickTorch Hyperparameter Search', add_help=False)
    parser.add_argument('--study-name', type=str, help='unique name for the hyptune experiment', default='hyptune')
    parser.add_argument('--config', type=str, help='path to json config', default='config/example_config.yaml')
    parser.add_argument('--no-ddp', action='store_true', help='disable DDP')
    parser.add_argument('--port', type=int, help='DDP port', default=None)
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--patience', type=int, help='patience for early stopping', default=-1)
    parser.add_argument('--model-path', type=str, help='ckpt path to continue', default=None)
    parser.add_argument('--no-save', action='store_true', help='disable logging and checkpoint saving (for debugging)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    storage_name = f'sqlite:///{args.study_name}.db'
    study = optuna.create_study(study_name=args.study_name, storage=storage_name, directions=['minimize'], load_if_exists=True)
    study.set_metric_names(['MSE'])
    study.optimize(wrapper_objective(args))
