
from argparse import ArgumentParser
from pathlib import Path
from prettytable import PrettyTable
import yaml

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--language_model', type=str, default='bert-base-uncased', help='language model for text embedding')
    parser.add_argument('--max_length', type=int, default=32, help='max length of the input text')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone of the model')
    parser.add_argument('--dataset', type=str, default='agiqa-1k', choices=['agiqa-1k', 'agiqa-3k'])
    parser.add_argument('--dataset_class', type=str, default='AGIDataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset, can be a soft link')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='path to cache directory, cache is for store temporary files')
    parser.add_argument('--label_name', type=str, default='mos', help='name of the label column in the csv file')
    parser.add_argument('--label_names', type=str, default='mos_q,mos_a', help='name of the label columns in the csv file, separated by comma, for multi-task learning, label_name will be ignored')
    parser.add_argument('--label_scale', type=float, default=1.0, help='scale of the label, for example, 0.05 for mos means 0-100 to 0-5')
    parser.add_argument('--stage', type=str, default='all', choices=['train', 'test', 'predict', 'all'], help='train/val or test')
    parser.add_argument('--run_name', type=str, default='resnet50-agiqa-1k', help='name of the run')
    parser.add_argument('--enable_transforms', action='store_true', help='enable default data augmentation for training data')
    parser.add_argument('--ckpt_path', type=str, default='best', help='path to checkpoint file, best or last? or the actual path')
    parser.add_argument('--fit_scale', action='store_true', help='fit scale when calculating metrics, seems to be only for statistic models, in our case, we use neural networks, and there is rarely a need to fit scale because the result metrics are almost the same')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='val data ratio')  # train,val,test 7:2:1, train-val then test
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--loss_func', type=str, default='mse', choices=['mse', 'mae', 'huber', 'smoothl1', 'fidelity', 'margin', 'margin_mse', 'ranknet', 'ranknet_mse', 'crossentropy'])
    parser.add_argument('--inference_gpu', type=int, default=0, help='gpu id for inference only')

    parser.add_argument('--precision', type=str, default='16-mixed', help='precision for training, for large models, we use 16-mixed, damn, bf16 losses some precision')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, default="[0]", help='list of gpus, [1,2] means card 1 and card 2')
    parser.add_argument('--cpu', action='store_true', help='use cpu for training')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file, override command line arguments')
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--wandbproject', type=str, default='agiqa', help='wandb project name')
    parser.add_argument('--pooler', type=str, default='mean', help='mean, max, last, firstlastmean')

    args = vars(parser.parse_args())

    # merge to config file
    if Path(args['config']).exists():
        with open(args['config'], 'r') as f:
            config = yaml.safe_load(f)
        if type(config) is dict:
            args.update(config)

    # print
    tab = PrettyTable(['Argument', 'Value'], align='l')
    for key, value in args.items():
        tab.add_row([key, value])
    print(tab)


    return args

def init_wandb(args):
    if args['wandb']:
        import wandb
        wandb.init(project=args['wandbproject'],
                   config=args, 
                   name=args['run_name'],
                   resume=args['stage'] == 'test')

