
from typing_extensions import deprecated
import torch
import torch.nn as nn
from torchvision.transforms import v2

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

@deprecated('Use scipy.optimize.curve_fit instead')
class FiveParamLogisticFunction(nn.Module):
    # not fit like this, use curve_fit to fit the parameters
    def __init__(self, init_params=None):
        super().__init__()
        
        # Initialize the 5 learnable parameters
        self.beta1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta3 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta4 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta5 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Initialize the parameters if provided
        if init_params is not None:
            self.beta1.data.fill_(init_params[0])
            self.beta2.data.fill_(init_params[1])
            self.beta3.data.fill_(init_params[2])
            self.beta4.data.fill_(init_params[3])
            self.beta5.data.fill_(init_params[4])

    def forward(self, x):
        """
        Applies the 5-parameter logistic function to the input tensor x.
        
        Args:
            x (torch.Tensor): Input tensor of predicted quality scores.
        
        Returns:
            torch.Tensor: Output tensor of mapped MOS scores.
        """
        return self.beta1 * (0.5 - 1 / (1 + torch.exp(self.beta2 * (x - self.beta3)))) + self.beta4 * x + self.beta5

def logistic_5_param(x, beta1, beta2, beta3, beta4, beta5):
    """
    Applies the 5-parameter logistic function to the input tensor x.
    
    Args:
        x (torch.Tensor): Input tensor of predicted quality scores.
        beta1 (float): The first learnable parameter.
        beta2 (float): The second learnable parameter.
        beta3 (float): The third learnable parameter.
        beta4 (float): The fourth learnable parameter.
        beta5 (float): The fifth learnable parameter.
    
    Returns:
        torch.Tensor: Output tensor of mapped MOS scores.
    """
    return beta1 * (0.5 - 1 / (1 + torch.exp(beta2 * (x - beta3)))) + beta4 * x + beta5

def resnet50transform(x):
    """
    Applies the transformation to the input tensor x. including normalization and resizing.
    
    Args:
        x (torch.Tensor): Input tensor of the image
    """
    x = v2.Compose([
        # v2.Resize(256),
        # v2.CenterCrop(224),
        # v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(x)

    return x

def map_y_to_quality(y):
    # y: float
    # map y to quality string:
    #   0-1: bad
    #   1-2: poor
    #   2-3: fair
    #   3-4: good
    #   4-5: excellent

    if y < 1:
        quality = 'bad'
    elif y < 2:
        quality = 'poor'
    elif y < 3:
        quality = 'fair'
    elif y < 4:
        quality = 'good'
    else:
        quality = 'excellent'
    return quality
