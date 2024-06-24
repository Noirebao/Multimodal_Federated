# Prototype Mask and Contrast for Multimodal FL
Official Implementation of paper "Multimodal Federated Learning with Missing Modality via Prototype Mask and Contrast". In this repository, we provide the implementation of the PmcmFL (**P**rototype **M**ask and **C**ontrast for **M**ultimodal **F**ederated **L**earning) algorithm.

![image](./figs/overview.png)

## Data Preparation

The VQAv2 dataset needs to be placed in the following directory:

```
-- src
---- data
------ train2014
        (images)
------ val2015
        (images)
------ test2015
        (images)
------ vqa
        (annotations)
```


## Training

Please first execute the following python command in the terminal to build data partition:
```
from datasets.dataset import VQAv2Dataset
from transformers import XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer("init_weight/beit3.spm")
VQAv2Dataset.make_dataset_index(data_path="./data", tokenizer=tokenizer, annotation_data_path="data/vqa", scale=False)
VQAv2Dataset.make_modal_missing_index('./data', modal_missing_rate=0.5, alpha=0.1, n_clients=30, n_classes=310)
```

To quickly perform training, run the following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 mian.py
```
Please refer to the main.py for more detailed training configuration.

## Testing
To test the trained global model, run the following command:
```
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=1 test_ckpt.py
```

## Acknowledgement
This repository is built using the [BEiT3](https://github.com/microsoft/unilm/tree/master/beit3) and the [CreamFL](https://github.com/FLAIR-THU/CreamFL) repository.
