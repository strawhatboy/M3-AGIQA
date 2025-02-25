
import json
from dataset import AGIDataset, Dataset
from wrapper import Wrapper
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.data as data
import torch
import torchvision as tv
from utils import init_wandb, set_seed, parse_args
from loguru import logger
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.strategies import DeepSpeedStrategy
from tqdm import tqdm
from pathlib import Path

if __name__ == '__main__':
    args = parse_args()
    set_seed(args['seed'])
    init_wandb(args)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_SRCC", mode="max",
        save_top_k=1,
        dirpath=f'./checkpoints/{args["model"]}/{args["run_name"]}',
        filename='best-{epoch:02d}-{val_SRCC:.2f}',
        save_last=True,
        )
    
    model = Wrapper(args)
    model.strict_loading = False    # so we can load the model without the huge frozen backbone
    datasetclass = model.model.get_dataset_class()
    train_data, val_data, test_data = datasetclass.load_data(args, model.model)

    logger.info(f'train data size: {len(train_data)}')
    logger.info(f'val data size: {len(val_data)}')
    logger.info(f'test data size: {len(test_data)}')

    train_data_loader = data.DataLoader(train_data, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True, collate_fn=model.model.collate_fn)
    val_data_loader = data.DataLoader(val_data, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False, collate_fn=model.model.collate_fn)
    test_data_loader = data.DataLoader(test_data, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False, collate_fn=model.model.collate_fn)
    if len(test_data_loader) <= 1:
        # use val_data_loader as test_data_loader if test_data_loader is empty, to test the best model
        test_data_loader = val_data_loader
    trainer_args = {
        'max_epochs': args['max_epochs'],
        'callbacks': [checkpoint_callback],
        'logger': WandbLogger() if args['wandb'] else False,  # do not log locally
        'accelerator': 'cuda',
        "precision": args['precision'], 
        'devices': eval(args['gpu']),
    }

    trainer = Trainer(**trainer_args)
    
    if args['stage'] == 'test':
        trainer.test(model, test_data_loader, ckpt_path=args['ckpt_path'])
        exit(0)
    elif args['stage'] == 'predict':
        predictions = trainer.predict(model, test_data_loader, ckpt_path=args['ckpt_path'])
        output_dict = {}    # key: image name, value: prediction
        for idx, batch in tqdm(enumerate(test_data_loader)):
            img_pathes = batch[0][0]
            if args['model'].lower().startswith('minicpm'):
                img_pathes = batch[0][1]
            # predictions = model(batch)
            for idx_inbatch, img_path in enumerate(img_pathes):
                result = predictions[idx]
                output_dict[Path(img_path).name] = result[idx_inbatch].detach().cpu().item()   # output probability
        
        # save the predictions
        predictions_run_filename = f'./predictions/{args["model"]}/{args["run_name"]}.json'
        # create the directory if not exists
        Path(predictions_run_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_run_filename, 'w') as f:
            json.dump(output_dict, f)
        exit(0)
    
    if args['ckpt_path'] != 'best':
        trainer.fit(model, train_data_loader, val_data_loader, ckpt_path=args['ckpt_path']) # resume from ckpt?
    else:
        trainer.fit(model, train_data_loader, val_data_loader)
    # best model path
    logger.info(f'best model path: {checkpoint_callback.best_model_path}')

    if args['stage'] == 'train':
        exit(0)
    trainer.test(model, test_data_loader, ckpt_path='best') # stage: all



