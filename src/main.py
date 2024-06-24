import os
import sys
sys.path.append(os.path.abspath('./datasets'))
sys.path.append(os.path.abspath('./model'))
sys.path.append(os.path.abspath('./optims'))
sys.path.append(os.path.abspath('./utils'))
sys.path.append(os.path.abspath('./algorithm'))
import argparse
import numpy as np
import random
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from algorithm.MMFL import MMFL
from utils import utils
import model.beit3_modeling  # This is important
import logging


def get_args():
    parser = argparse.ArgumentParser()
    # Model Parameters
    parser.add_argument('--model', type=str, default='beit3_base_patch16_480')
    parser.add_argument('--task', type=str, default='vqav2')
    parser.add_argument('--input_size', type=int, default=480)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--checkpoint_activations', action='store_true', default=None)
    parser.add_argument('--sentencepiece_model', type=str, default='./init_weight/beit3.spm')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--log_dir', default='./log')
    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=True)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)
    # result parameters
    parser.add_argument('--output_dir', default='./save_weight/scale')
    parser.add_argument('--log_path', default='./save_weight/scale/log.txt')
    # important optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    # Federated parameters
    parser.add_argument('--local_epochs', default=5, type=int)              # 3
    parser.add_argument('--communication_rounds', default=20, type=int)     # 30
    parser.add_argument('--client_nums', default=35, type=int)              # 30
    parser.add_argument('--selected_client_nums', default=10, type=float)    # 5
    # top parameters5
    parser.add_argument('--v_loss_weight', default=1e-8, type=float)
    parser.add_argument('--l_loss_weight', default=1e-8, type=float)
    parser.add_argument('--f_loss_weight', default=0.01, type=float)
    parser.add_argument('--mse_loss_weight', default=0.05*768, type=float)  # 0.05*768
    parser.add_argument('--kd_loss_weight', default=0.05, type=float)
    parser.add_argument('--regularization_start_round', default=2, type=int)  # 2
    parser.add_argument('--kd_temperature', default=0.5, type=float)
    parser.add_argument('--fedprox_weight', default=1.0, type=float)
    parser.set_defaults(modal_missing=False)
    parser.set_defaults(prototype_as_missing_modal=False)
    parser.set_defaults(prototype_as_rep_target=False)
    parser.set_defaults(img_proto_target=False)
    parser.set_defaults(text_proto_target=False)
    parser.set_defaults(fusion_proto_target=False)
    parser.set_defaults(fediot=False)
    parser.set_defaults(fedpac=False)
    parser.set_defaults(fedhkd=False)
    parser.add_argument('--modal_missing', action='store_true', default=False)
    parser.add_argument('--prototype_as_missing_modal', action='store_true', default=False)
    parser.add_argument('--prototype_as_rep_target', action='store_true', default=False)
    parser.add_argument('--no_img_proto_target', action='store_true', default=False)
    parser.add_argument('--no_text_proto_target', action='store_true', default=False)
    parser.add_argument('--no_fusion_proto_target', action='store_true', default=False)
    parser.add_argument('--fediot', action='store_true', default=False)
    parser.add_argument('--fedpac', action='store_true', default=False)
    parser.add_argument('--fedhkd', action='store_true', default=False)
    parser.add_argument('--fedprox', action='store_true', default=False)

    # other
    parser.set_defaults(embed_dim=768)
    parser.set_defaults(num_classes=310)

    return parser.parse_args()


def main(args):
    # init distributed training
    utils.init_distributed_mode(args)

    logger = utils.Logger(log_path=args.log_path)
    logger.write("create logger.")

    # init setting
    args.total_epochs = args.communication_rounds * args.local_epochs
    device = torch.device(args.device)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False

    # get train split
    train_split_list_img_text = []
    train_split_list_img_only = []
    train_split_list_text_only = []
    split_file_name = None
    if not args.modal_missing:
        split_file_name = "modal-missing-non-iid"
        for idx in range(args.client_nums):
            split_name = "modalmissing-noniid" + "." + split_file_name + "." + "client%d-img-text" % idx
            train_split_list_img_text.append(split_name)
    else:
        split_file_name = "modal-missing-non-iid"
        for idx in range(args.client_nums):
            split_name = "modalmissing-noniid" + "." + split_file_name + "." + "client%d-img-text" % idx
            train_split_list_img_text.append(split_name)
        for idx in range(args.client_nums):
            split_name = "modalmissing-noniid" + "." + split_file_name + "." + "client%d-img-only" % idx
            train_split_list_img_only.append(split_name)
        for idx in range(args.client_nums):
            split_name = "modalmissing-noniid" + "." + split_file_name + "." + "client%d-text-only" % idx
            train_split_list_text_only.append(split_name)

    # get sample num for each client
    sample_num_dict = {}
    statistic_file_path = os.path.join(args.data_path, split_file_name, "statistic.jsonl")
    with open(statistic_file_path, mode="r", encoding="utf-8") as reader:
        for line in reader:
            line_split = line.split(" ")
            client_id = int(line_split[0].split(":")[0][6:])
            total_sample_nums = int(line_split[0].split(":")[1])
            img_text_sample_nums = int(line_split[1].split(":")[1])
            img_only_sample_nums = int(line_split[2].split(":")[1])
            text_only_sample_nums = int(line_split[3].split(":")[1])
            both_missing_sample_nums = int(line_split[4].split(":")[1])
            sample_num_dict[client_id] = {
                "total": total_sample_nums, "img-text": img_text_sample_nums, "img-only": img_only_sample_nums,
                "text-only": text_only_sample_nums, "both-missing": both_missing_sample_nums
            }

    # federated learning
    torch.distributed.barrier()
    algo = MMFL(args=args, client_nums=args.client_nums, sample_num_dict=sample_num_dict,
                device=device, val_split="val",
                train_split_list_img_text=train_split_list_img_text,
                train_split_list_img_only=train_split_list_img_only,
                train_split_list_text_only=train_split_list_text_only,
                logger=logger,
                )

    # start training
    train_flag = True
    if train_flag:
        torch.distributed.barrier()
        algo.run(
            n_comm_rounds=args.communication_rounds,
            selected_client_nums=args.selected_client_nums,
        )


if __name__ == '__main__':

    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)




