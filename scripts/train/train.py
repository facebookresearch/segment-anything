#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2022-05-12
@Author : nhhung1810
@File   : train_orig.py

NOTE:
This is a refactored version of models/ARMN/train.py.
"""
import os
import sys
from datetime import datetime
from loguru import logger
from typing import Tuple
import loguru
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sympy import FlagError
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch

# Internal
from scripts.train.flare22_loader import FLARE22
from scripts.train.loss import MultimaskSamLoss
from scripts.train.sam_train import SamTrain
from scripts.utils import summary
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry

NAME = "sam_simple_obj_train"
TIME = datetime.now().strftime("%y%m%d-%H%M%S")
ex = Experiment(NAME)

# NOTE: Refactoring
logger.remove()
logger.add(
    sys.stdout,
    format="\n<lvl>[{time:DD:MMM:YY HH:mm:ss}] - [{level}] - {message}</lvl>",
)


@ex.config
def config():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: the effective batch-size will = batch_size * gradient_accumulation_step
    batch_size = 4
    logdir = f"runs/{NAME}-{TIME}"
    resume_iteration = None
    n_epochs = 100

    # NOTE
    gradient_accumulation_step = 1

    # Model params
    focal_gamma = 2.0
    focal_alpha = None
    # label_smoothing = 0.1
    # sequence_length = 327680
    # model_complexity = 16
    # model_complexity_lstm = 16
    # validation_length = sequence_length

    # Optim params
    learning_rate = 6e-4
    learning_rate_decay_steps = 10000 * gradient_accumulation_step
    learning_rate_decay_rate = 0.98
    clip_gradient_norm = 3

    ex.observers.append(FileStorageObserver.create(logdir))
    pass


@ex.capture
def make_dataset(batch_size) -> Tuple[FLARE22, DataLoader]:
    FLARE22.LIMIT = 20
    dataset = FLARE22(is_debug=True, is_save_gpu=False)
    dataset.preprocess()
    dataset.self_check()
    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    return loader


@ex.capture
def make_model(
    device,
    logdir,
    resume_iteration,
    learning_rate,
    learning_rate_decay_steps,
    learning_rate_decay_rate,
    focal_gamma,
    focal_alpha,
) -> Tuple[SamTrain, Optimizer, StepLR, int]:
    def load_model(checkpoint="./sam_vit_b_01ec64.pth", checkpoint_type="vit_b") -> Sam:
        sam: Sam = sam_model_registry[checkpoint_type](checkpoint=checkpoint)
        return sam

    model = load_model()
    model.to(device=device)

    if resume_iteration is None:
        logger.warning("Fresh-new model is initialized")
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    # else:
    #     model_path = os.path.join(logdir, f"model-{resume_iteration}.pt")
    #     logger.success(f"Load pre-trained model at {model_path}")
    #     model.load_state_dict(torch.load(model_path))
    #     optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # if isinstance(label_smoothing, float):
    #     model.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    #     logger.success(f"Enable label smoothing of {label_smoothing}")
    #     pass

    logger.info("Pretty print")
    summary(model)

    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate
    )

    sam_train = SamTrain(sam_model=model)

    loss_fnc = MultimaskSamLoss(
        reduction="mean", focal_alpha=focal_alpha, focal_gamma=focal_gamma
    )

    return sam_train, optimizer, scheduler, loss_fnc


@ex.automain
def train(
    logdir,
    device,
    n_epochs,
    clip_gradient_norm,
    gradient_accumulation_step,
):
    print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    loader = make_dataset()
    sam_train, optimizer, scheduler, loss_fnc = make_model()

    assert isinstance(sam_train, SamTrain), ""
    assert isinstance(optimizer, Optimizer), ""
    assert isinstance(scheduler, StepLR), ""
    assert isinstance(loss_fnc, MultimaskSamLoss), ""

    optimizer.zero_grad()
    loop = tqdm(range(0, n_epochs), total=n_epochs, desc="Training...")
    for i in loop:
        for batch in tqdm(loader, desc=f"Epoch {i}", leave=False):
            img_emb: Tensor = batch["img_emb"]
            mask: Tensor = batch["mask"]
            input_size = batch["input_size"]
            original_size = batch["original_size"]
            masks_pred, iou_pred, _ = sam_train.predict_torch(
                image_emb=img_emb,
                input_size=(input_size[0, 0], input_size[0, 1]),
                original_size=(original_size[0, 0], original_size[0, 1]),
                multimask_output=True,
                return_logits=True,
            )

            mask = mask.unsqueeze(1).repeat_interleave(3, dim=1).type(torch.int64)

            loss = loss_fnc.forward(
                multi_mask_pred=masks_pred,
                multi_iou_pred=iou_pred,
                multi_mask_target=mask,
            )

            # Normalize loss
            loss: Tensor = loss / gradient_accumulation_step
            loss.backward()

            # This will update the gradient at once
            if i % gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                # NOTE: clear the gradient
                optimizer.zero_grad()

            #  This will prevent the gradient from explosion
            if clip_gradient_norm:
                clip_grad_norm_(sam_train.model.parameters(), clip_gradient_norm)

            writer.add_scalar("train/loss", loss.item(), global_step=i)
