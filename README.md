# QuickTorch
PyTorch framework to handle all boilerplate code for training deep learning models. This framework is built with research in mind and is designed to be easily adopted so you can focus simply on handling the actual model architetecture and dataloading. 

This code is **battle-tested** and has been used in almost all [my publications and winning submissions](https://steveimm.id/) in various competitions.

Main features:
- Flexible, modular, and transparent
- Support distributed training
- Support logging, checkpointing
- Multiple validation phases in a single epoch
- Deterministic training
- Automatic hyperparameter search

## Installation
Install all dependencies by running:
```bash
pip install -r requirements.txt
```

## Code Structure
### Data
In the `data` folder, you can define the dataset for your task. Create a class that inherits from `torch.utils.data.Dataset` and implement the `__getitem__` and `__len__` methods. See [data/example_data.py](data/example_data.py) for an example.

### Model
In the `model` folder is where you implement the model architecture. Create a class that inherits from `torch.nn.Module` and implement the `forward` method. See [model/example_model.py](model/example_model.py) for an example.

### Trainer
This is the most complex part. Trainer handles all the training loop including validation, logging, and the distributed training support. 
In the `trainer` folder, you can find [trainer/base_trainer.py](trainer/base_trainer.py) which is the base class for the trainer. 

`BaseTrainer` is the parent class all trainer should inherit from. You need to implement the following methods:
- `_get_scheduler` to return the learning rate scheduler
- `_get_loss_fn` to return the loss function
- `step` to define one iteration of the training loop. Inside this method, you should define the forward pass, calculate the loss, and return the loss tensor.

All other methods are self-explanatory and can be overridden if needed.

Additionally, there is also `Tracker` class which takes care of tracking the metric during training. Importantly you can set the `direction` parameter to `max` or `min` to indicate whether the metric should be maximized or minimized.

See [trainer/example_trainer.py](trainer/example_trainer.py) for an example.

### Config
Config is defined inside a `.yaml` file. In this file the following parameters must exist:
```yaml
model:
  <define all model related parameters here>

train:
  uid: <the unique id for the run>
  lr: <learning rate>
  log_dir: <directory to save logs>
  eval_per_epoch: <number of evaluation per epoch>
  patience: <number of epochs to wait before early stopping>
  epoch: <number of epochs to train>
  batch_size: <batch size>
  n_workers: <number of workers for dataloader>
  ckpt_interval: <number of epochs between saving checkpoints>

data:
  <define all data related parameters here>
```

## Training
Training script can be derived from [train.py](train.py). You need to modify the `model`, `trainer`, `train_dataset`, and `val_dataset` to your implementation. To run the training, simply run:
```bash
python train.py --config <path to yaml config file> --uid <unique id for the run>
```

By default, the training script will utilize all available GPUs. If you want to run on some particular GPUs, you can set the `CUDA_VISIBLE_DEVICES` environment variable. 

On the other hand, if you want to only run in a single GPU, add a `--no-ddp` flag. This will disable the distributed training and make the training faster (if dataloading is the bottleneck) because it does not need to configure communications, sampling, and synching between GPUs.

To disable logging and checkpoint to the log directory, add a `--no-save` flag. This is useful for debugging so that you don't create multiple log files.

## Hyperparameter Search
You need to configure which hyperparameters to search and all the candidate values in the script. The script can be run by:
```bash
python hyptune.py --config <path to yaml config file> --study-name <name of the study>
```

See [hyptune.py](hyptune.py) for an example.

To see the results of the hyperparameter search, run:
```bash
optuna-dashboard sqlite:///<name of the study>.db --host 0.0.0.0
```
## Logging
Tensorboard is utilized for logging. To see the logs, run:
```bash
tensorboard --logdir <path to log directory>
```

## References
- [PyTorch DDP tutorials](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj)
- [DDP vs DP](https://huggingface.co/docs/transformers/en/perf_train_gpu_many)