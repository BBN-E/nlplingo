import torch
import torch.nn as nn
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *
import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer


def get_pool_mask(i, triggers, seq_len):
    batch_size = triggers.shape[0]
    triggers = triggers.data.cpu().numpy()
    pool_mask = []
    for t_id in range(batch_size):
        j = triggers[t_id]
        min_id = min(i, j)
        max_id = max(i, j)
        mask = [1 if k not in set(range(min_id, max_id + 1)) else 0 for k in range(seq_len)]
        mask = torch.Tensor(mask).bool()
        pool_mask.append(mask)
    pool_mask = torch.stack(pool_mask, dim=0)  # [batch size, seq len]
    return pool_mask


def max_pooling3d(mask, reps):
    '''
    take-out positions are marked -1, padding positions are marked 0
    mask.shape = [batch size, seq len]
    reps.shape = [batch size, seq len, rep dim]
    max pooling along axis 1
    '''
    mask = mask.long().eq(0).bool()
    pool = reps.masked_fill_(mask.unsqueeze(2), -INFINITY_NUMBER)
    pool = torch.max(pool, dim=1)[0]
    return pool


def max_pooling2d(mask, reps):
    '''
    mask.shape = [seq len]
    reps.shape = [seq len, rep dim]
    max pooling along axis 0
    '''
    mask = mask.long().eq(0).bool()
    pool = reps.masked_fill_(mask.unsqueeze(1), -INFINITY_NUMBER)
    pool = torch.max(pool, dim=0)[0]
    return pool


### class
class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                        weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) * \
                               init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)

                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss


### torch specific functions
def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name in ['adagrad', 'myadagrad']:
        # use my own adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2)  # use default lr
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2)  # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad


### model IO
def save(model, optimizer, opt, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")


def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    opt = dump['config']
    return model, optimizer, opt


def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']


def move_to_device(batch, device):
    new_batch = [b.to(device) for b in batch]
    return new_batch


def get_edge_reps(h, head_ids, pad_masks, device):
    '''
        h.shape = [batch size, seq len, rep dim]
        pad_masks = [batch size, seq len]
    '''
    batch_size, seq_len, rep_dim = h.shape

    root = torch.zeros(batch_size, 1, rep_dim).to(device)
    h_root = torch.cat([h, root], dim=1)  # [batch size, 1 + seq len, rep dim]

    head_reps = torch.stack([h_root[sent_id][head_ids[sent_id]] for sent_id in range(batch_size)],
                            dim=0)  # [batch size, seq len, rep dim]
    tail_reps = h_root[:, 1:, :]  # [batch size, seq len, rep dim]
    edge_reps = torch.cat([head_reps, tail_reps], dim=2)  # [batch size, seq len, 2 * rep dim]

    input_masks = pad_masks.long().eq(0).bool()

    edge_reps = edge_reps.float() * input_masks.unsqueeze(2).float()
    return edge_reps
