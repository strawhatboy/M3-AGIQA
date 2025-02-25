
import torch
import torch.nn as nn
import math



class FidelityLoss(nn.Module):
    def __init__(self, esp=1e-6):
        super().__init__()
        self.esp = esp

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        B = y_pred.size(0)
        # it is a ranking loss
        # the diff of every two in y_pred
        expanded = y_pred.expand(B, -1)
        diff = (expanded - expanded.T) / math.sqrt(2)
        # p_hat is diff after standard Normal cumulative distribution function
        p_hat = 0.5 * (1 + torch.erf(diff))

        # p is boolean, in y_true which one is larger than the other
        expanded_y_true = y_true.expand(B, -1)
        p = (expanded_y_true - expanded_y_true.T) >= 0
        p = p.float()

        # loss: 1-sqrt(p*p_hat) - sqrt((1-p)*(1-p_hat))
        loss = 1 - torch.sqrt(p * p_hat + self.esp) - torch.sqrt((1 - p) * (1 - p_hat) + self.esp)
        return torch.mean(loss)

class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, margin=1.0):
        # Reshape y_true and y_pred to column vectors
        y_true = y_true.view(-1, 1)
        y_pred = y_pred.view(-1, 1)
        
        # Compute the pairwise differences
        true_diff = y_true - y_true.t()
        pred_diff = y_pred - y_pred.t()
        
        # Create a mask for positive differences in true values
        mask = (true_diff > 0).float()
        
        # Compute the pairwise ranking loss
        loss = torch.clamp(margin - pred_diff * mask, min=0)
        
        # Average the loss over the number of valid pairs
        num_pairs = mask.sum()
        if num_pairs > 0:
            loss = loss.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        
        return loss

# margin loss 4 times the mse loss
class MarginMSECombinedLoss(nn.Module):
    def __init__(self, margin=0.001, mse_weight=0.8):
        super().__init__()
        self.margin = margin
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        self.margin_loss = MarginLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mse_loss = self.mse_loss(y_pred, y_true)
        margin_loss = self.margin_loss(y_pred, y_true, self.margin)
        return mse_loss * self.mse_weight + margin_loss * (1 - self.mse_weight)


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_true = y_true.view(-1, 1)
        y_pred = y_pred.view(-1, 1)
        
        # Compute pairwise differences
        pred_diff = y_pred - y_pred.t()
        true_diff = y_true - y_true.t()
        
        # Convert true differences to binary targets
        y_diff = (true_diff > 0).float()
        
        # Apply the sigmoid function to the predicted differences
        P_ij = torch.sigmoid(pred_diff)
        
        # Compute the RankNet loss
        loss = - y_diff * torch.log(P_ij) - (1 - y_diff) * torch.log(1 - P_ij)
        
        # Average the loss over all pairs
        num_pairs = y_diff.numel()
        loss = loss.sum() / num_pairs
        
        return loss

class RankNetMSECombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.8):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        self.ranknet_loss = RankNetLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mse_loss = self.mse_loss(y_pred, y_true)
        ranknet_loss = self.ranknet_loss(y_pred, y_true)
        return mse_loss * self.mse_weight + ranknet_loss * (1 - self.mse_weight)
