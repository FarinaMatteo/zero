import os
import time
import random
import builtins

import numpy as np
import pandas as pd

from enum import Enum
from datetime import timedelta

import torch
import torch.distributed as dist


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            builtins.print(*args, **kwargs)
    else:
        builtins.print(*args, **kwargs)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, fabric=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)
        
    def display_summary(self, prefix="Summary: "):
        print(f"\n{prefix}")
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries), end="\n\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device != 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def display_results(results, save_to='', decimals=2):
    print("======== Result Summary ========")
    paths = []
    for set_id, results_dict in results.items():
        set_dataframe = {"set_id": [set_id]}
        for k, v in results_dict.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    set_dataframe[k] = [v.item()]
            else:
                set_dataframe[k] = [v]
        
        set_dataframe = pd.DataFrame.from_dict(set_dataframe)
        set_dataframe = pd.DataFrame(set_dataframe).round(decimals=decimals)
        print(set_dataframe.to_string(index=False), end="\n")

        if save_to:
            dirname, filename = os.path.split(save_to)
            path_for_this_set = os.path.join(dirname, f"{set_id}_{filename}")
            set_dataframe.to_csv(path_for_this_set, index=False)
            paths.append(path_for_this_set)

        iterables_dataframe = {}
        max_length = 0
        for k, v in results_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                iterables_dataframe[k] = v.tolist()
            elif isinstance(v, list):
                iterables_dataframe[k] = v
            if k in iterables_dataframe and len(iterables_dataframe[k]) > max_length:
                max_length = len(iterables_dataframe[k]) 
        
        if len(iterables_dataframe) > 0:
            iterables_dataframe["set_id"] = [set_id] * max_length
            iterables_dataframe = pd.DataFrame.from_dict(iterables_dataframe)
            iterables_dataframe = pd.DataFrame(iterables_dataframe).round(decimals=decimals)
            if save_to:
                path_for_this_set = os.path.join(dirname, f"{set_id}_iterables_{filename}")
                iterables_dataframe.to_csv(path_for_this_set, index=False)

    print("================================")
    
    for path in paths:
        print(f"Results saved to: {path}")
    
    return


def arg_in_results(results, key, arg):
    for set_id, results_dict in results.items():
        if key not in results_dict:
            results_dict[key] = arg
    return results


def break_sample_tie(ties, logit, device):
    ties = torch.tensor(ties, dtype=torch.int, device=device)
    logit[~ties] = -torch.inf
    scalar_pred = torch.argmax(logit, dim=-1)
    return scalar_pred


def greedy_break(ties, logits, device):
    ties_tensor = torch.tensor(ties, dtype=torch.int, device=device)
    preds = torch.argmax(logits, dim=1)
    for pred in preds:
        if pred in ties_tensor:
            return pred
    return break_sample_tie(ties, logit=logits[0], device=device)
