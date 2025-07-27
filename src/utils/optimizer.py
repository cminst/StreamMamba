""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import re
import torch
from torch import optim as optim
from utils.distributed import is_main_process
import logging
logger = logging.getLogger(__name__)
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay, no_decay_list=(), filter_bias_and_bn=True):
    named_param_tuples = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
            named_param_tuples.append([name, param, 0])
        elif name in no_decay_list:
            named_param_tuples.append([name, param, 0])
        else:
            named_param_tuples.append([name, param, weight_decay])
    return named_param_tuples


def add_different_lr(named_param_tuples_or_model, diff_lr_names, diff_lr, default_lr, log = False):
    """use lr=diff_lr for modules named found in diff_lr_names,
    otherwise use lr=default_lr

    Args:
        named_param_tuples_or_model: List([name, param, weight_decay]), or nn.Module
        diff_lr_names: List(str)
        diff_lr: float
        default_lr: float
        log: bool
    Returns:
        named_param_tuples_with_lr: List([name, param, weight_decay, lr])
    """
    named_param_tuples_with_lr = []
    logger.info(f"diff_names: {diff_lr_names}, diff_lr: {diff_lr}")
    for name, p, wd in named_param_tuples_or_model:
        use_diff_lr = False
        for diff_name in diff_lr_names:
            if re.search(diff_name, name) is not None:
                if log:
                    logger.info(f"param {name} use different_lr: {diff_lr}")
                use_diff_lr = True
                break

        lr_to_use = diff_lr if use_diff_lr else default_lr
        named_param_tuples_with_lr.append(
            [name, p, wd, lr_to_use]
        )

    if is_main_process() and log:
        for name, _, wd, diff_lr in named_param_tuples_with_lr:
            logger.info(f"param {name}: wd: {wd}, lr: {diff_lr}")

    return named_param_tuples_with_lr


def create_optimizer_params_group(named_param_tuples_with_lr, default_lr):
    """named_param_tuples_with_lr: List([name, param, weight_decay, lr])"""
    # Group parameters by name
    param_groups = {}

    for name, param, wd, lr in named_param_tuples_with_lr:
        key = f"{name}_{wd}_{lr}"
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "weight_decay": wd,
                "lr": lr,
                "name": name,
                "different_lr": (lr != default_lr)  # Flag if this is a different lr
            }
        param_groups[key]["params"].append(param)

    # Convert to list of dictionaries for optimizer
    optimizer_params_group = list(param_groups.values())

    # Log the parameter groups
    for group in optimizer_params_group:
        logger.info(f"optimizer -- name={group['name']} lr={group['lr']} wd={group['weight_decay']} params_count={len(group['params'])}")

    return optimizer_params_group


def create_optimizer(args, model, filter_bias_and_bn=True, return_group=False):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # check for modules that requires different lr
    if hasattr(args, "different_lr") and args.different_lr.enable:
        diff_lr_module_names = args.different_lr.module_names
        diff_lr = args.different_lr.lr
    else:
        diff_lr_module_names = []
        diff_lr = None

    no_decay = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay = model.no_weight_decay()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, 'no_weight_decay'):
            no_decay = model.module.no_weight_decay()
            no_decay = {"module." + k for k in no_decay}

    named_param_tuples = add_weight_decay(
        model, weight_decay, no_decay, filter_bias_and_bn)
    named_param_tuples = add_different_lr(
        named_param_tuples, diff_lr_module_names, diff_lr, args.lr)
    parameters = create_optimizer_params_group(named_param_tuples, args.lr)

    if return_group:
        return parameters

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer


def extend_optimizer_with_param_groups(optimizer, scheduler, param_groups):
    """Add parameter groups to an existing optimizer and update scheduler.

    Args:
        optimizer (Optimizer): The optimizer to extend.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Scheduler tied
            to the optimizer. ``base_lrs`` will be extended if provided.
        param_groups (List[dict]): Parameter groups returned by
            :func:`create_optimizer_params_group`.
    """

    for group in param_groups:
        add_group = {k: group[k] for k in ["params", "lr", "weight_decay"]}
        optimizer.add_param_group(add_group)
        # attach custom info for logging
        optimizer.param_groups[-1]["name"] = group.get("name")
        optimizer.param_groups[-1]["different_lr"] = group.get("different_lr", False)

        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs.append(group["lr"])
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr.append(group["lr"])
