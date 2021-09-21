import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ste_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, init_weight_sorted, perm_size):
        ctx.save_for_backward(init_weight_sorted)
        if weight.numel() % perm_size == 0:
            # reshape and then sort weight along dim=1
            col_idx = torch.argsort(weight.view(-1, perm_size)).view(-1)
            # row indices
            row_idx = perm_size * torch.arange(weight.view(-1, perm_size).size(0)).repeat_interleave(perm_size).to(
                device)
            # assign values
            weight.view(-1)[col_idx+row_idx] = init_weight_sorted
        else:
            n_row = weight.numel() // perm_size
            weight_dividable = weight.view(-1)[:n_row*perm_size].view(-1, perm_size)
            weight_rest = weight.view(-1)[n_row*perm_size:]
            # sort weight_dividable along dim=1
            col_idx_dividable = torch.argsort(weight_dividable).view(-1)
            col_idx_rest = torch.argsort(weight_rest)
            col_idx = torch.cat([col_idx_dividable, col_idx_rest], dim=0)
            # row indices
            row_idx = perm_size * torch.arange(n_row+1).repeat_interleave(perm_size).to(device)[:weight.numel()]
            # assign values
            weight.view(-1)[col_idx+row_idx] = init_weight_sorted
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class LinearQuantized(nn.Linear):
    def __init__(self, in_features, out_features, perm_size=16, bias=True):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.perm_size = perm_size
        if self.weight.numel() % self.perm_size == 0:
            self.init_weight_sorted = torch.sort(self.weight.view(-1, self.perm_size))[0].view(-1)
        else:
            n_row = self.weight.numel() // self.perm_size
            self.init_weight_sorted = torch.zeros(self.weight.numel()).to(device)
            self.init_weight_sorted[:n_row * self.perm_size] = torch.sort(self.weight.view(-1)[:n_row * self.perm_size].view(-1, self.perm_size))[
                0].view(-1)
            self.init_weight_sorted[n_row * self.perm_size:] = torch.sort(self.weight.view(-1)[n_row * self.perm_size:])[0]

    def set_init_weight_sorted(self):
        if self.weight.numel() % self.perm_size == 0:
            self.init_weight_sorted = torch.sort(self.weight.view(-1, self.perm_size))[0].view(-1)
        else:
            n_row = self.weight.numel() // self.perm_size
            self.init_weight_sorted = torch.zeros(self.weight.numel()).to(device)
            self.init_weight_sorted[:n_row * self.perm_size] = \
            torch.sort(self.weight.view(-1)[:n_row * self.perm_size].view(-1, self.perm_size))[
                0].view(-1)
            self.init_weight_sorted[n_row * self.perm_size:] = \
            torch.sort(self.weight.view(-1)[n_row * self.perm_size:])[0]

    def get_init_weight_sorted(self):
        return self.init_weight_sorted

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.init_weight_sorted.clone().detach().to(device), self.perm_size)
        return F.linear(x, weight, self.bias)


class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, perm_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.perm_size = perm_size
        if self.weight.numel() % self.perm_size == 0:
            self.init_weight_sorted = torch.sort(self.weight.view(-1, self.perm_size))[0].view(-1)
        else:
            n_row = self.weight.numel() // self.perm_size
            self.init_weight_sorted = torch.zeros(self.weight.numel()).to(device)
            self.init_weight_sorted[:n_row * self.perm_size] = \
            torch.sort(self.weight.view(-1)[:n_row * self.perm_size].view(-1, self.perm_size))[
                0].view(-1)
            self.init_weight_sorted[n_row * self.perm_size:] = \
            torch.sort(self.weight.view(-1)[n_row * self.perm_size:])[0]

    def set_init_weight_sorted(self):
        if self.weight.numel() % self.perm_size == 0:
            self.init_weight_sorted = torch.sort(self.weight.view(-1, self.perm_size))[0].view(-1)
        else:
            n_row = self.weight.numel() // self.perm_size
            self.init_weight_sorted = torch.zeros(self.weight.numel()).to(device)
            self.init_weight_sorted[:n_row * self.perm_size] = \
                torch.sort(self.weight.view(-1)[:n_row * self.perm_size].view(-1, self.perm_size))[
                    0].view(-1)
            self.init_weight_sorted[n_row * self.perm_size:] = \
                torch.sort(self.weight.view(-1)[n_row * self.perm_size:])[0]

    def get_init_weight_sorted(self):
        return self.init_weight_sorted

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.init_weight_sorted.clone().detach().to(device), self.perm_size)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
