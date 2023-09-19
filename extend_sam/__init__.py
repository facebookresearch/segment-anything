# copyright ziqi-jin
import torch
from .extend_sam import BaseExtendSam, SemanticSam
from .runner import BaseRunner, SemRunner
# from .optimizer import BaseOptimizer
from .scheduler import WarmupMultiStepLR
from .utils import get_opt_pamams

AVAI_SCH = ["single_step", "multi_step", "warmup_multi_step", "cosine", "linear"]
AVAI_MODEL = {'base_sam': BaseExtendSam, 'sem_sam': SemanticSam}
# AVAI_OPT = {'base_opt': BaseOptimizer, 'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}
AVAI_OPT = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
AVAI_RUNNER = {'base_runner': BaseRunner, 'sem_runner': SemRunner}


def get_model(model_name, **kwargs):
    if model_name not in AVAI_MODEL:
        print('not supported model name, please implement it first.')
    return AVAI_MODEL[model_name](**kwargs).cuda()


def get_optimizer(opt_name, **kwargs):
    if opt_name not in AVAI_OPT:
        print('not supported optimizer name, please implement it first.')
    return AVAI_OPT[opt_name](**{k: v for k, v in kwargs.items() if v is not None})


def get_runner(runner_name):
    if runner_name not in AVAI_RUNNER:
        print('not supported runner name, please implement it first.')
    return AVAI_RUNNER[runner_name]


def get_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=1,
        gamma=0.1,
        warmup_factor=0.01,
        warmup_steps=10,
        max_epoch=1,
        n_epochs_init=50,
        n_epochs_decay=50,

):
    """A function wrapper for building a learning rate scheduler.
    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is
            single_step.
        stepsize (int or list, optional): step size to decay learning rate.
            When ``lr_scheduler`` is "single_step", ``stepsize`` should be an integer.
            When ``lr_scheduler`` is "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.
    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = get_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = get_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCH
            )
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                "be an integer, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "warmup_multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(
                "For warmup multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=stepsize,
            gamma=gamma,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_steps,
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, int(max_epoch)
        )

    elif lr_scheduler == "linear":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - n_epochs_init) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule
        )

    return scheduler
