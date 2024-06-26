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

    ckpt_path = "./save_weight/server/global_model-30.pth"
    
    utils.load_model_and_may_interpolate(
        ckpt_path=ckpt_path, model=model, model_key='model|module', model_prefix=''
    )

    evaluate(data_loader_val, model, device, task_handler)



