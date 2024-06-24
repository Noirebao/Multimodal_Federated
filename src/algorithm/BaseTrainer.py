import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEma
from utils import utils
from torch.distributed.algorithms.join import Join


class VQAHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None
        self.predictions = []
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.ClipLossHalf = utils.ClipLossHalf(rank=utils.get_rank(), world_size=utils.get_world_size())
        self.clip_logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.MSELoss = nn.MSELoss(reduction="mean")
        self.KDLoss = nn.MSELoss(reduction="sum")
        self.label2ans = None

    def train_batch(self, args, model, prototypes, mode, image, language_tokens, padding_mask, labels):

        model.module.set_mode(mode)

        if mode == "vl":
            logits, hidden_reps = model(image=image, question=language_tokens, padding_mask=padding_mask)
            ce_loss = self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1]
            clip_loss_img, clip_loss_text, clip_loss_fusion = None, None, None
            if args.prototype_as_rep_target:
                index = torch.argmax(labels, dim=1).detach()
                clip_loss_img = self.ClipLossHalf(hidden_reps["img_rep"], prototypes["img"][index],
                                                  self.clip_logits_scale.exp())
                clip_loss_text = self.ClipLossHalf(hidden_reps["text_rep"], prototypes["text"][index],
                                                   self.clip_logits_scale.exp())
                clip_loss_fusion = self.ClipLossHalf(hidden_reps["fusion_rep"], prototypes["fusion"][index],
                                                     self.clip_logits_scale.exp())
            return {
                "ce_loss": ce_loss,
                "clip_loss_img": clip_loss_img, "clip_loss_text": clip_loss_text, "clip_loss_fusion": clip_loss_fusion,
            }

        elif mode == "v":
            text_prototypes = torch.randn([args.batch_size, args.embed_dim], requires_grad=False, dtype=torch.float32)
            if args.prototype_as_missing_modal:
                assert len(prototypes) > 0
                index = torch.argmax(labels, dim=1).detach()
                text_prototypes = prototypes["text"][index]
            logits, hidden_reps = model(image=image, question=None, padding_mask=None,
                                        prototype=text_prototypes)
            ce_loss = self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1]
            clip_loss_img, clip_loss_text, clip_loss_fusion = None, None, None
            if args.prototype_as_rep_target:
                index = torch.argmax(labels, dim=1).detach()
                clip_loss_img = self.ClipLossHalf(hidden_reps["img_rep"], prototypes["img"][index],
                                                  self.clip_logits_scale.exp())
                clip_loss_fusion = self.ClipLossHalf(hidden_reps["fusion_rep"], prototypes["fusion"][index],
                                                     self.clip_logits_scale.exp())
            return {
                "ce_loss": ce_loss,
                "clip_loss_img": clip_loss_img, "clip_loss_text": clip_loss_text, "clip_loss_fusion": clip_loss_fusion,
            }

        elif mode == "l":
            img_prototypes = torch.randn([args.batch_size, args.embed_dim], requires_grad=False, dtype=torch.float32)
            if args.prototype_as_missing_modal:
                assert len(prototypes) > 0
                index = torch.argmax(labels, dim=1).detach()
                img_prototypes = prototypes["img"][index]
            logits, hidden_reps = model(image=None, question=language_tokens, padding_mask=padding_mask,
                                        prototype=img_prototypes)
            ce_loss = self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1]
            clip_loss_img, clip_loss_text, clip_loss_fusion = None, None, None
            if args.prototype_as_rep_target:
                index = torch.argmax(labels, dim=1).detach()
                clip_loss_text = self.ClipLossHalf(hidden_reps["text_rep"], prototypes["text"][index],
                                                   self.clip_logits_scale.exp())
                clip_loss_fusion = self.ClipLossHalf(hidden_reps["fusion_rep"], prototypes["fusion"][index],
                                                     self.clip_logits_scale.exp())
            return {
                "ce_loss": ce_loss,
                "clip_loss_img": clip_loss_img, "clip_loss_text": clip_loss_text, "clip_loss_fusion": clip_loss_fusion,
            }

        else:
            raise NotImplementedError("not implemented for mode %s" % mode)

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split
        self.predictions.clear()
        self.metric_logger = metric_logger
        self.label2ans = data_loader.dataset.label2ans

    def eval_batch(self, model, image, language_tokens, padding_mask, labels=None, qid=None):
        if isinstance(model, torch.nn.Module):
            model.set_mode("vl")
        else:
            model.module.set_mode("vl")
        logits, _ = model(image=image, question=language_tokens, padding_mask=padding_mask)
        batch_size = language_tokens.shape[0]
        if labels is not None:
            scores = utils.VQAScore()(logits, labels) * 100.0
            self.metric_logger.meters['score'].update(scores.item(), n=batch_size)
        else:
            _, preds = logits.max(-1)
            for image_id, pred in zip(qid, preds):
                self.predictions.append({
                    "question_id": image_id.item(),
                    "answer": self.label2ans[pred.item()],
                })

    def after_eval(self, **kwargs):
        if len(self.predictions) == 0:
            print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
            return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"
        else:
            return self.predictions, "prediction"

    def eval_batch_image_only(self, model, image, prototypes, labels, qid=None):
        if isinstance(model, torch.nn.Module):
            model.set_mode("v")
        else:
            model.module.set_mode("v")
        logits, _ = model(image=image, question=None, padding_mask=None, prototype=prototypes)
        batch_size = labels.shape[0]
        scores = utils.VQAScore()(logits, labels) * 100.0
        self.metric_logger.meters['score'].update(scores.item(), n=batch_size)

    def eval_batch_text_only(self, model, text, padding, prototypes, labels, qid=None):
        if isinstance(model, torch.nn.Module):
            model.set_mode("l")
        else:
            model.module.set_mode("l")
        logits, _ = model(image=None, question=text, padding_mask=padding, prototype=prototypes)
        batch_size = labels.shape[0]
        scores = utils.VQAScore()(logits, labels) * 100.0
        self.metric_logger.meters['score'].update(scores.item(), n=batch_size)


def train_one_epoch(
        args, model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: VQAHandler, global_epoch: int, local_epoch: int,
        start_steps: int, lr_schedule_values: list, loss_scaler,
        prototypes, max_norm: float = 0, update_freq: int = 1,
        model_ema: Optional[ModelEma] = None, mode: str = "vl",
        logger=None
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Global Epoch: [{}]'.format(global_epoch)
    print_freq = 1
    total_steps = len(data_loader)

    model.module.set_mode(mode)
    logger.write("model mode switch to %s" % mode)

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step

        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = args.lr * param_group["lr_scale"]

        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            results = handler.train_batch(args, model, prototypes, mode, **data)

        ce_loss = results.pop("ce_loss")
        ce_loss_value = ce_loss.item()

        clip_loss_img = results.pop("clip_loss_img")
        clip_loss_text = results.pop("clip_loss_text")
        clip_loss_fusion = results.pop("clip_loss_fusion")
        clip_loss_img_value = np.nan
        clip_loss_text_value = np.nan
        clip_loss_fusion_value = np.nan
        if args.img_proto_target:
            if mode == "vl" or mode == "v":
                clip_loss_img_value = clip_loss_img.item()
        if args.text_proto_target:
            if mode == "vl" or mode == "l":
                clip_loss_text_value = clip_loss_text.item()
        if args.fusion_proto_target:
            clip_loss_fusion_value = clip_loss_fusion.item()

        loss = ce_loss
        if args.img_proto_target:
            if mode == "vl" or mode == "v":
                loss += clip_loss_img * args.v_loss_weight
        if args.text_proto_target:
            if mode == "vl" or mode == "l":
                loss += clip_loss_text * args.l_loss_weight
        if args.fusion_proto_target:
            loss += clip_loss_fusion * args.f_loss_weight

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order  # False
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(ce_loss=ce_loss_value)
        metric_logger.update(clip_loss_i=clip_loss_img_value)
        metric_logger.update(clip_loss_t=clip_loss_text_value)
        metric_logger.update(clip_loss_f=clip_loss_fusion_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        logger.write("global_epoch{%d} local_epoch{%d} [%d/%d] min_lr=%.6f lr=%.6f "
                     "loss=%.6f ce=%.6f clip_i=%.6f clip_t=%.6f clip_f=%.6f loss_scale=%.2f"
                     " weight_decay=%.4f gn=%.6f" % (global_epoch, local_epoch, step, total_steps, min_lr, max_lr,
                                                     loss_value, ce_loss_value, clip_loss_img_value,
                                                     clip_loss_text_value, clip_loss_fusion_value,
                                                     loss_scale_value, weight_decay_value, grad_norm)
                     )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logger.write("Averaged stats:")
    logger.write(metric_logger)


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()


