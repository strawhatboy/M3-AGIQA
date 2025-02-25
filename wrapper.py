import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as metrics
import torchvision.models as models
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np

from loss import FidelityLoss, MarginLoss, MarginMSECombinedLoss, RankNetLoss, RankNetMSECombinedLoss
from models.MiniCPM import MiniCPMIQA, MiniCPMIQA_GRUHeader, MiniCPMIQA_GRUHeaderResidual, MiniCPMIQA_LSTMHeader, MiniCPMIQA_Transformer, MiniCPMIQA_mambaHeader, MiniCPMIQA_nofinetune, MiniCPMIQA_xLSTMHeader
from models.utils import logistic_5_param
from loguru import logger
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

class MetricsCalculator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pearsonr = metrics.PearsonCorrCoef()
        self.spearmanr = metrics.SpearmanCorrCoef()
        self.kendallr = metrics.KendallRankCorrCoef()
        self.args = args

    def fit_curve(self, y_hat, y):
        beta = [10, 0.1, torch.mean(y).item(), 0.1, 0.1]
        popt, _ = curve_fit(logistic_5_param, y_hat, y, p0=beta)
        y_logistic = logistic_5_param(y_hat, *popt)
        return y_logistic

    def forward(self, y_hat, y):
        if self.args['fit_scale']:
            y_logistic = self.fit_curve(y_hat, y)
            plcc = pearsonr(y_logistic, y)[0]
            srcc = spearmanr(y_logistic, y)[0]
            krcc = kendalltau(y_logistic, y)[0]
        else:
            plcc = self.pearsonr(y_hat, y)
            srcc = self.spearmanr(y_hat, y)
            krcc = self.kendallr(y_hat, y)
        return plcc, srcc, krcc

class Wrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = self.select_model(args)
        # metrics_calculator should be embedded inside the model, but not in the Wrapper
        # because some model would calculate metrics for multiple tasks
        # which would return SRCC(quality) and SRCC(alignment) for multiple labels for example
        # for single task, return SRCC is ok.
        self.metrics_calculator = MetricsCalculator(args)   
        self.args = args
        # calculate pearsonr and spearmanr on y_hat and y at the end of each epoch
        self.y_hats = []
        self.ys = []

        self.test_y_hats = []
        self.test_ys = []

        self.loss_func = self.get_loss_func()

        logger.info(f'using model: \n{self.model}')

    def forward(self, batch):
        x, y = batch
        return self.model(x, y)
    
    def get_supported_models(self):
        return ['resnet50', 'stairiqa-resnet50', 'mobilenet_v2', 'mgqa', 'resnet50-fr', 'mambaiqa']
    
    def select_model(self, args):
        model_name = args['model']
        if model_name == 'MiniCPMIQA_nofinetune':
            return MiniCPMIQA_nofinetune(args)
        elif model_name == 'minicpm-xlstm':
            return MiniCPMIQA_xLSTMHeader(args)
        elif model_name == 'minicpm-lstm':
            return MiniCPMIQA_LSTMHeader(args)
        elif model_name == 'minicpm-gru':
            return MiniCPMIQA_GRUHeader(args)
        elif model_name == 'minicpm-gru-residual':
            return MiniCPMIQA_GRUHeaderResidual(args)
        elif model_name == 'minicpm-mamba':
            return MiniCPMIQA_mambaHeader(args)
        elif model_name == 'minicpm-trm':
            return MiniCPMIQA_Transformer(args)
        else:
            raise ValueError('model_name not supported')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x, y)
        if hasattr(self.model, 'loss_func'):
            loss = self.model.loss_func(y_hat, y)
        else:
            loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss.mean(), sync_dist=True)
        # self.log('train_accuracy', 1-loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x, y)
        if hasattr(self.model, 'loss_func'):
            loss = self.model.loss_func(y_hat, y) 
        else:
            loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss.mean(), sync_dist=True)
        if hasattr(self.model, 'forward_y_hat'):
            y_hat = self.model.forward_y_hat(y_hat)
        # self.log('val_accuracy', 1-loss)
        if self.args['fit_scale']:
            # put on cpu
            self.y_hats.append(y_hat.clone().detach().cpu())
            self.ys.append(y.clone().detach().cpu())
        else:
            self.y_hats.append(y_hat)
            self.ys.append(y)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        if self.args['optimizer'] == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer'] == 'sgd':
            opt = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer'] == 'adamw':
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer'] == 'ds_fusedadam':
            opt = FusedAdam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer'] == 'ds_cpuadam':
            opt = DeepSpeedCPUAdam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        
        return opt
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x, y)
        if hasattr(self.model, 'loss_func'):
            loss = self.model.loss_func(y_hat, y)
        else:
            loss = self.loss_func(y_hat, y)
        self.log('test_loss', loss, sync_dist=True)

        if hasattr(self.model, 'forward_y_hat'):
            y_hat = self.model.forward_y_hat(y_hat)
        # self.log('test_accuracy', 1-loss)
        if self.args['fit_scale']:
            self.test_y_hats.append(y_hat.clone().detach().cpu())
            self.test_ys.append(y.clone().detach().cpu())
        else:
            self.test_y_hats.append(y_hat)
            self.test_ys.append(y)
        return loss

    def on_validation_epoch_end(self) -> None:
        # calculate pearsonr and spearmanr on y_hat and y at the end of each epoch
        y_hat = torch.cat(self.y_hats)
        y = torch.cat(self.ys)
        
        with torch.no_grad():
            plcc, srcc, krcc = self.metrics_calculator(y_hat, y)
            self.log('val_PLCC', plcc, sync_dist=True)
            self.log('val_SRCC', srcc, sync_dist=True)
            self.log('val_KRCC', krcc, sync_dist=True)

        self.y_hats = []
        self.ys = []

    def on_test_epoch_end(self) -> None:
        # calculate pearsonr and spearmanr on y_hat and y at the end of each epoch
        y_hat = torch.cat(self.test_y_hats)
        y = torch.cat(self.test_ys)
        
        with torch.no_grad():
            plcc, srcc, krcc = self.metrics_calculator(y_hat, y)
            self.log('test_PLCC', plcc, sync_dist=True)
            self.log('test_SRCC', srcc, sync_dist=True)
            self.log('test_KRCC', krcc, sync_dist=True)

        self.test_y_hats = []
        self.test_ys = []

    def get_loss_func(self):
        if self.args['loss_func'] == 'mse':
            return F.mse_loss
        elif self.args['loss_func'] == 'mae':
            return F.l1_loss
        elif self.args['loss_func'] == 'huber':
            return F.smooth_l1_loss
        elif self.args['loss_func'] == 'logcosh':
            return F.smooth_l1_loss
        elif self.args['loss_func'] == 'fidelity':
            return FidelityLoss()
        elif self.args['loss_func'] == 'margin':
            return MarginLoss()
        elif self.args['loss_func'] == 'margin_mse':
            return MarginMSECombinedLoss()
        elif self.args['loss_func'] == 'ranknet':
            return RankNetLoss()
        elif self.args['loss_func'] == 'ranknet_mse':
            return RankNetMSECombinedLoss()
        elif self.args['loss_func'] == 'crossentropy':
            return F.cross_entropy
        else:
            return F.mse_loss
        
    def on_save_checkpoint(self, checkpoint):
        # remove the freezed model inside model, key start with model.model
        if self.args['freeze_backbone']:
            for key in list(checkpoint['state_dict'].keys()):
                if key.startswith('model.model.'):
                    del checkpoint['state_dict'][key]
        pass

