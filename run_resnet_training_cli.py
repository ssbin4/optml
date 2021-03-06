from ast import Store, parse
from torch.utils.data import DataLoader
from resnet import get_pretrained_resnet
import torch
from resnet import PalmNet
import logging
from resnet_stanford_cars_cli import StanfordCarsCLI, Hyperparameters, CLI

import torch
import random
import torch.backends.cudnn as cudnn

import numpy as np


class ResnetTrainingCLI(StanfordCarsCLI):
    def __init__(self):
        super().__init__()

    def arg_parse(self):
        """Adding Hyperparameters to CLI arguments"""
        parser = super().arg_parse()
        parser.add_argument("--" + Hyperparameters.SCEDULER_RATE.value, dest=Hyperparameters.SCEDULER_RATE.value, type=float,
                            help="number of epochs to wait before annealing learning rate (when using reduce and step)", required=True)
        parser.add_argument("--" + Hyperparameters.LEARNING_RATE.value, dest=Hyperparameters.LEARNING_RATE.value, type=float,
                            help="learning rate to use", required=True)
        parser.add_argument("--" + Hyperparameters.BATCH_SIZE.value, dest=Hyperparameters.BATCH_SIZE.value, type=int,
                            help="batch size to use", required=True)
        parser.add_argument("--" + Hyperparameters.LEARNING_RATE_SCHEDULER.value, dest=Hyperparameters.LEARNING_RATE_SCHEDULER.value, type=float,
                            help="annealing schedule rate to use. multiplied to learning rate", required=True)
        parser.add_argument("--" + Hyperparameters.WEIGHT_DECAY.value, dest=Hyperparameters.WEIGHT_DECAY.value, type=float,
                            help = "weight decay to use", required=True)
        parser.add_argument("--" + Hyperparameters.LR_SCHEDULER.value, dest=Hyperparameters.LR_SCHEDULER.value, type=str,
                            help = "learning scheduler to experiment (added)", choices=['const', 'step', 'exp', 'cos', 'reduce'], required=True)
        parser.add_argument("--gamma", dest='gamma', type=float,
                            help = "gamma value used in exponential decay", default=0.95)
        parser.add_argument("--t_0", dest='t_0', type=int,
                            help = "t_0 value used in cosine annealing decay. Number of iterations for the first restart", default=1)
        parser.add_argument("--t_mult", dest='t_mult', type=int,
                            help = "t_mult value used in cosine annealing decay", default=1)
        parser.add_argument("--eta_min", dest='eta_min', type=float,
                            help = "eta_min used in exponential decay", default=0.001)
        parser.add_argument("--fc_init_lr", dest='fc_init_lr', type=float,
                            help = "initial lr value for final FC layer when applying different initial learning rate", default=0.01)
        parser.add_argument("--" + Hyperparameters.MOMENTUM.value, dest=Hyperparameters.MOMENTUM.value, type=float,
                            help = "momentum to use", required=True)
        parser.add_argument("--" + Hyperparameters.NESTEROV.value, dest=Hyperparameters.NESTEROV.value, action='store_true',
                            help="use Nesterov")
        parser.add_argument("--" + "no-"+ Hyperparameters.NESTEROV.value, dest=Hyperparameters.NESTEROV.value, action='store_false',
                            help="do not use Nesterov")
        parser.add_argument("--diff_lr", dest='diff_lr', action='store_true', help="Use different initial lr for pretrained network and final FC layer")
        parser.add_argument("--same_lr", dest="diff_lr", action='store_false', help="Use same initial lr for pretrained network and final FC layer")
        return parser

    def load_datasets(self, parsed_cli_arguments):
        return super().load_datasets(parsed_cli_arguments)

    def run(self, parsed_cli_arguments, training_data, validation_data):

        logging.info("loading pretrained model and establishing model characteristics")

        resnet_pretrained_model = get_pretrained_resnet(parsed_cli_arguments[CLI.FREEZE_WEIGHTS.value],
                                                        parsed_cli_arguments[CLI.NUM_CLASSES.value],
                                                        parsed_cli_arguments[CLI.MODEL.value])
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        
        if parsed_cli_arguments['diff_lr']:
            fc_list = ['fc.weight', 'fc.bias']
            base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in fc_list, resnet_pretrained_model.named_parameters()))))
            fc_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in fc_list, resnet_pretrained_model.named_parameters()))))
            sgd_optimizer = torch.optim.SGD([{'params': base_params, 'lr': parsed_cli_arguments[Hyperparameters.LEARNING_RATE.value]},
                                            {'params': fc_params, 'lr': parsed_cli_arguments['fc_init_lr']}
                                            ],
                                        momentum=parsed_cli_arguments[Hyperparameters.MOMENTUM.value],
                                        weight_decay=parsed_cli_arguments[Hyperparameters.WEIGHT_DECAY.value],
                                        nesterov=parsed_cli_arguments[Hyperparameters.NESTEROV.value])
        else:
            sgd_optimizer = torch.optim.SGD(resnet_pretrained_model.parameters(),
                                        lr=parsed_cli_arguments[Hyperparameters.LEARNING_RATE.value],
                                        momentum=parsed_cli_arguments[Hyperparameters.MOMENTUM.value],
                                        weight_decay=parsed_cli_arguments[Hyperparameters.WEIGHT_DECAY.value],
                                        nesterov=parsed_cli_arguments[Hyperparameters.NESTEROV.value])

        lr_scheduler =  parsed_cli_arguments[Hyperparameters.LR_SCHEDULER.value] 
        if lr_scheduler == 'reduce':
            learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min',
                                                                             factor=parsed_cli_arguments[Hyperparameters.LEARNING_RATE_SCHEDULER.value],
                                                                             patience=parsed_cli_arguments[Hyperparameters.SCEDULER_RATE.value],
                                                                             verbose=True)
        elif lr_scheduler == 'const':
            learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer, parsed_cli_arguments[CLI.EPOCHS.value], gamma=1.0)
        elif lr_scheduler == 'step':
            learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer, parsed_cli_arguments[Hyperparameters.SCEDULER_RATE.value], gamma=parsed_cli_arguments[Hyperparameters.LEARNING_RATE_SCHEDULER.value])
        elif lr_scheduler == 'exp':
            learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(sgd_optimizer, gamma=parsed_cli_arguments['gamma'])
        elif lr_scheduler == 'cos':
            learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(sgd_optimizer, T_0=parsed_cli_arguments['t_0'], T_mult=parsed_cli_arguments['t_mult'], eta_min=parsed_cli_arguments['eta_min'])

        logging.info("training model")
        palm_net = PalmNet(epochs=parsed_cli_arguments[CLI.EPOCHS.value], gd_optimizer=sgd_optimizer, model=resnet_pretrained_model,
                           loss_function=cross_entropy_loss,
                           learning_rate_scheduler=learning_rate_scheduler,
                           validation_frequency=parsed_cli_arguments[CLI.VALIDATION_FREQUENCY.value],
                           torch_checkpoint_location=parsed_cli_arguments[CLI.CHECKPOINT.value],
                           model_checkpointing=parsed_cli_arguments[CLI.CHECKPOINT_FREQUENCY.value],
                           lr_scheduler=parsed_cli_arguments[Hyperparameters.LR_SCHEDULER.value],
                           init_lr=parsed_cli_arguments[Hyperparameters.LEARNING_RATE.value],
                           gamma=parsed_cli_arguments['gamma'],
                           t_0=parsed_cli_arguments['t_0'],
                           t_mult=parsed_cli_arguments['t_mult'],
                           eta_min=parsed_cli_arguments['eta_min'],
                            annealing_factor=parsed_cli_arguments[Hyperparameters.LEARNING_RATE_SCHEDULER.value],
                            scheduler_rate=parsed_cli_arguments[Hyperparameters.SCEDULER_RATE.value],
                            freeze_weights=parsed_cli_arguments[CLI.FREEZE_WEIGHTS.value],
                            diff_lr = parsed_cli_arguments['diff_lr'],
                            fc_init_lr = parsed_cli_arguments['fc_init_lr']
                            )

        trained_model, validation_metric = palm_net.train_model(training_data=DataLoader(training_data,
                                                                                         batch_size=parsed_cli_arguments[Hyperparameters.BATCH_SIZE.value],
                                                                                         shuffle=True),
                                                                validation_data=DataLoader(validation_data,
                                                                                           batch_size=parsed_cli_arguments[Hyperparameters.BATCH_SIZE.value],
                                                                                           shuffle=True),
                                                                number_of_labels=parsed_cli_arguments[
                                                                    CLI.NUM_CLASSES.value])
        return trained_model, validation_metric


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    resnet_training_cli = ResnetTrainingCLI()
    resnet_training_cli.run_all()
