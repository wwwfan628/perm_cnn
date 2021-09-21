import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@torch.no_grad()
def prune_weight_abs(param, amount=0.9):
    thr = (len(param.view(-1)) - 1) * amount
    param.view(-1)[torch.argsort(param.abs().view(-1)) < thr] = 0


@torch.no_grad()
def prune_weights_abs(params, amount=0.9):
    params = list(params)
    params_abs_flatten = np.zeros(0)
    params_shape = []
    params_start_idx = []
    param_start_idx = 0
    for param in params:
        params_abs_flatten = np.append(params_abs_flatten, param.abs().view(-1).clone().detach().cpu())
        params_shape.append(param.shape)
        params_start_idx.append(param_start_idx)
        param_start_idx += len(param.view(-1))
    params_abs_flatten = torch.Tensor(params_abs_flatten).to(device)
    k = int(len(params_abs_flatten) * (1-amount))
    idx_topk = torch.topk(params_abs_flatten, k=k)[1]
    masks_flatten = torch.zeros(params_abs_flatten.shape).to(device)
    masks_flatten[idx_topk] = 1
    for i, param in enumerate(params):
        if i < (len(params)-1):
            mask = masks_flatten[params_start_idx[i]:params_start_idx[i+1]].reshape(params_shape[i])
        else:
            mask = masks_flatten[params_start_idx[i]:].reshape(params_shape[i])
        param.mul_(mask)
