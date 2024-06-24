import torch
from main import get_args
from utils import utils
import numpy as np
import random
from datasets.dataset import create_dataset_by_split
from timm.models import create_model
from algorithm.BaseTrainer import evaluate, VQAHandler


if __name__ == '__main__':

    args = get_args()
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    data_loader_val = create_dataset_by_split(args, split="val", is_train=False)

    task_handler = VQAHandler()

    model = create_model(
        "beit3_base_patch16_480_vqav2",
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations
    )

    model.to(device)

    ckpt_path = "./save_weight/scale/server/global_model-9.pth"  # 6
    # ckpt_path = "./exp-res/noniid0.1-missmodal0.5/baseline3-pac/server/global_model-14.pth"

    # CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port='29510' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --master_port='29511' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='2' python -m torch.distributed.launch --master_port='29512' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='3' python -m torch.distributed.launch --master_port='29513' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='4' python -m torch.distributed.launch --master_port='29514' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='5' python -m torch.distributed.launch --master_port='29515' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='6' python -m torch.distributed.launch --master_port='29516' --nproc_per_node=1 test_ckpt.py
    # CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port='29517' --nproc_per_node=1 test_ckpt.py

    utils.load_model_and_may_interpolate(
        ckpt_path=ckpt_path, model=model, model_key='model|module', model_prefix=''
    )

    evaluate(data_loader_val, model, device, task_handler)



