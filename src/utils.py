import torch
import torch.nn as nn
from torch.utils import data
from scipy import sparse

def get_device(i = 1):
    if torch.cuda.device_count() >= i +1:
        return torch.device(f"cuda:{i}")
    else:
        return torch.device("cpu")
    
def get_TF(file_path, adata, target_index = None):
    '''Get TF list from file_path'''
    with open(file_path, 'r') as f:
        tf_rows = f.readlines()
        tf_list = [tf.strip() for tf in tf_rows]
    tf_index = []
    for tf in tf_list:
        if tf in adata.var_names:
            tf_index.append(adata.var_names.get_loc(tf))
    if target_index:
        tf_index.append(target_index)
    print(tf_index)
    tf_adata = adata[:, tf_index]
    return tf_adata

def data_loader(adata, batch_size, shuffle = False):
    if sparse.issparse(adata.X):
        dat = adata.X.A
    else:
        dat = adata.X
    dat = torch.Tensor(dat)
    dataloader = data.DataLoader(dat, batch_size =batch_size, shuffle = shuffle) # type: ignore
    return dataloader

class HurdleLoss(nn.BCEWithLogitsLoss):
    '''
    Hurdle loss that incorporates ZCELoss for each output, as well as MSE for
    each output that surpasses the threshold value. This can be understood as
    the negative log-likelihood of a hurdle distribution.

    Args:
      lam: weight for the ZCELoss term (the hurdle).
      thresh: threshold that an output must surpass to be considered turned on.
    '''
    def __init__(self, lam=10.0, thresh=0):
        super().__init__()
        self.lam = lam
        self.thresh = thresh

    def forward(self, pred, target):
        # Verify prediction shape.
        if pred.shape[1] != 2 * target.shape[1]:
            raise ValueError(
                'Predictions have incorrect shape! For HurdleLoss, the'
                ' predictions must have twice the dimensionality of targets'
                ' ({})'.format(target.shape[1] * 2))

        # Reshape predictions, get distributional.
        pred = pred.reshape(*pred.shape[:-1], -1, 2)
        pred = pred.permute(-1, *torch.arange(len(pred.shape))[:-1])
        mu = pred[0]
        p_logit = pred[1]

        # Calculate loss.
        zero_target = (target <= self.thresh).float().detach()
        hurdle_loss = super().forward(p_logit, zero_target)
        mse = (1 - zero_target) * (target - mu) ** 2

        loss = self.lam * hurdle_loss + mse
        return torch.mean(torch.sum(loss, dim=-1))
