import json
import random
import torch
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform
import sys
import os
sys.path.append(os.path.abspath('./datasets'))
sys.path.append(os.path.abspath('./'))
from glossary import normalize_word
from randaug import RandomAugment
from utils import utils


def build_transform(is_train, args):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0),
                                              interpolation=args.train_interpolation),
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self, data_path, split, transform,
            tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


class VQAv2Dataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        ans2label_file = os.path.join(data_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = i
                label2ans.append(ans)

        self.ans2label = ans2label
        self.label2ans = label2ans

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return "vqa.train.jsonl", "vqa.trainable_val.jsonl"
        elif split == "val":
            return "vqa.rest_val.jsonl",
        elif split == "test":
            return "vqa.test.jsonl",
        elif split == "test-dev":
            return "vqa.test-dev.jsonl",
        elif split.startswith("noniid"):
            file_dir, file_name = split.split(".")[1], split.split(".")[2]
            return os.path.join(file_dir, file_name + '.jsonl'),
        elif split.startswith("modalmissing-noniid"):
            file_dir, file_name = split.split(".")[1], split.split(".")[2]
            return os.path.join(file_dir, file_name + '.jsonl'),
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        if "labels" in self.items[index] and len(self.items[index]["labels"]) > 0:
            labels = [0.] * len(self.label2ans)
            for l, s in zip(self.items[index]["labels"], self.items[index]["scores"]):
                labels[l] = s
            data["labels"] = torch.FloatTensor(labels)
        else:
            data["qid"] = self.items[index]["qid"]
        return data

    @staticmethod
    def get_score(occurences):
        if occurences == 0:
            return 0.0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1.0

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, annotation_data_path, scale=False, seed=42):
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]

        with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        for split, questions in zip(
                ["train", "val", "test", "test-dev"],
                [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                tokens = tokenizer.tokenize(question_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert q["question_id"] not in _annot[q["image_id"]]
                _annot[q["image_id"]][q["question_id"]] = {
                    "question": question_text,
                    "token_ids": token_ids,
                }

            annotations[split] = _annot

        all_major_answers = list()

        for split, annots in zip(
                ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            # _annot = annotations[split]
            for q in annots:
                all_major_answers.append(q["multiple_choice_answer"])

        all_major_answers = [normalize_word(word) for word in all_major_answers]

        if scale:
            counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}  # 3129 labels will be selected
        else:
            counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 180}  # 310 labels will be selected

        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())

        for split, annots in zip(
                ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q in annots:
                answers = q["answers"]
                answer_count = {}
                for answer in answers:
                    answer_ = answer["answer"]
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in ans2label:
                        continue
                    labels.append(ans2label[answer])
                    score = cls.get_score(answer_count[answer])
                    scores.append(score)

                assert "labels" not in _annot[q["image_id"]][q["question_id"]]
                assert "question" in _annot[q["image_id"]][q["question_id"]]
                _annot[q["image_id"]][q["question_id"]]["labels"] = labels
                _annot[q["image_id"]][q["question_id"]]["scores"] = scores

        for split in ["train", "val"]:
            filtered_annot = dict()
            for ik, iv in annotations[split].items():
                new_q = dict()
                for qk, qv in iv.items():
                    if len(qv["labels"]) != 0:
                        new_q[qk] = qv
                if len(new_q) != 0:
                    filtered_annot[ik] = new_q
            annotations[split] = filtered_annot

        split2items = {}
        for split in ["train", "val", "test", "test-dev"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014",
                "val": "val2014",
                "test": "test2015",
                "test-dev": "test2015",
            }[split]
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            random.seed(seed)
            random.shuffle(paths)
            annot_paths = [path for path in paths \
                           if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                print("all images have caption annotations")
            else:
                print("not all images have caption annotations")
            print(len(paths), len(annot_paths), len(annot))

            items = []
            for path in annot_paths:
                img_id = int(path.split("/")[-1].split("_")[-1][:-4])
                _annot = annotations[split][img_id]
                for q_id in _annot:
                    q = _annot[q_id]
                    if split in ["train", "val"]:
                        labels = q["labels"]
                        scores = q["scores"]
                    else:
                        labels, scores = [], []

                    items.append({
                        "image_path": os.path.join(split_name, path.split('/')[-1]),
                        "text_segment": q["token_ids"],
                        "labels": labels,
                        "scores": scores,
                        "qid": q_id,
                    })
            split2items[split] = items

        if scale:
            print("scale is True")
            for split in ["train", "val", "test", "test-dev"]:
                _write_data_into_jsonl(items=split2items[split], jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

            # Following ViLT, we use 1000 images of the original val set as the final val set
            val_image2items = defaultdict(list)
            for item in split2items["val"]:
                val_image2items[item["image_path"]].append(item)

            print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

            val_images = list(val_image2items.keys())
            random.seed(seed)
            random.shuffle(val_images)
            trainable_val = []
            rest_val = []
            for i, image_id in enumerate(val_images):
                if i < 1000:
                    rest_val += val_image2items[image_id]
                else:
                    trainable_val += val_image2items[image_id]

            _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
            _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

        else:
            print("scale is False")
            random.seed(seed)

            # 1/10 train items will be select
            train_items = split2items["train"]
            random.shuffle(train_items)
            train_items = train_items[:43000]
            _write_data_into_jsonl(items=train_items, jsonl_file=os.path.join(data_path, "vqa.train.jsonl"))

            # 1/10 val items will be select
            val_items = split2items["val"]
            random.shuffle(val_items)
            trainable_val_items = val_items[:21000]
            rest_val_items = val_items[21000:]
            _write_data_into_jsonl(items=trainable_val_items, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))

            for split in ["test", "test-dev"]:
                _write_data_into_jsonl(items=split2items[split], jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

            # use 5000 items in rest_val as the final val set
            random.seed(2 * seed)
            random.shuffle(rest_val_items)
            _write_data_into_jsonl(items=rest_val_items[:5000], jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

        # write answer2label
        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans,
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))

    @classmethod
    def make_noniid_index(cls, data_path, alpha, n_clients, n_classes, split="train", seed=42):
        items_files = cls.get_index_files(split=split)
        items = []
        offset = 0
        for _items_files in items_files:
            items_file = os.path.join(data_path, _items_files)
            with open(items_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, items_file))
                offset = len(items)
        random.seed(seed)
        # print(len(items))  # 639757

        np.random.seed(seed)
        dirichlet_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

        items_split_by_classes = defaultdict(list)
        for item in items:
            max_score_label_pos = np.argmax(item['scores'])
            max_score_label = item['labels'][max_score_label_pos]
            items_split_by_classes[max_score_label].append(item)

        print("start non-iid split ...")
        items_split_by_clients = defaultdict(list)
        for i_class in range(n_classes):
            i_class_nums = len(items_split_by_classes[i_class])
            if i_class_nums == 0:
                continue
            else:
                nums_for_clients = np.trunc(dirichlet_distribution[i_class] * i_class_nums)
                nums_for_clients[-1] = i_class_nums - sum(nums_for_clients[:-1])
                random.shuffle(nums_for_clients)
                assert sum(nums_for_clients) == i_class_nums
                for i_client, i_client_nums in enumerate(nums_for_clients):
                    for i in range(int(i_client_nums)):
                        items_split_by_clients[i_client].append(items_split_by_classes[i_class].pop(0))

        for i_client in items_split_by_clients.keys():
            random.shuffle(items_split_by_clients[i_client])
        print("finish non-iid split")

        # get write path
        noniid_file = os.path.join(data_path, f"non-iid")
        Path(noniid_file).mkdir(parents=True, exist_ok=True)

        # write sample numbers of each client into statistic.jsonl
        data_statistic = {}
        for _client_id in items_split_by_clients.keys():
            data_statistic[_client_id] = (len(items_split_by_clients[_client_id]))
        data_statistic_file_path = os.path.join(noniid_file, f"statistic.jsonl")
        with open(data_statistic_file_path, mode="w", encoding="utf-8") as writer:
            for _client_id in data_statistic.keys():
                writer.write("client%d:" % int(_client_id))
                writer.write(str(data_statistic[_client_id]))
                writer.write(" img-text:%d img-only:0 text-only:0 both-missing:0" % data_statistic[_client_id])
                writer.write('\n')
        print("finish non-iid data statistic")

        # write sample index of each client into client*.jsonl
        for _client_id in items_split_by_clients.keys():
            _items_per_client = items_split_by_clients[_client_id]
            jsonl_file_path = os.path.join(noniid_file, f"client{_client_id}.jsonl")
            with open(jsonl_file_path, mode="w", encoding="utf-8") as writer:
                for item in _items_per_client:
                    writer.write(json.dumps(item, indent=None))
                    writer.write('\n')
            print(f"finish write client{_client_id}'s items into json file")

        return items_split_by_clients, data_statistic

    @classmethod
    def make_modal_missing_index(cls, data_path, modal_missing_rate, alpha, n_clients, n_classes, split="train", seed=42):
        np.random.seed(seed)
        items_split_by_clients, data_statistic = cls.make_noniid_index(data_path=data_path, alpha=alpha,
                                                                       n_clients=n_clients, n_classes=n_classes,
                                                                       split=split, seed=seed)
        split_file_name = "modal-missing-non-iid"
        print("start modal-missing split ...")
        Path(os.path.join(data_path, split_file_name)).mkdir(parents=True, exist_ok=True)
        modal_statistic = {}
        for client_id in range(n_clients):
            img_missing = np.random.binomial(n=1, p=modal_missing_rate, size=data_statistic[client_id])
            text_missing = np.random.binomial(n=1, p=modal_missing_rate, size=data_statistic[client_id])
            items_list = items_split_by_clients[client_id]
            items_img_text = []
            items_img_only = []
            items_text_only = []
            # missing modal
            for idx in range(len(items_list)):
                if img_missing[idx] == 0 and text_missing[idx] == 0:
                    items_img_text.append(items_list[idx])
                elif img_missing[idx] == 1 and text_missing[idx] == 0:
                    items_text_only.append(items_list[idx])
                elif img_missing[idx] == 0 and text_missing[idx] == 1:
                    items_img_only.append(items_list[idx])
                else:
                    continue
            # save item index
            items_img_text_path = os.path.join(data_path, split_file_name, "client%d-img-text.jsonl" % client_id)
            items_img_only_path = os.path.join(data_path, split_file_name, "client%d-img-only.jsonl" % client_id)
            items_text_only_path = os.path.join(data_path, split_file_name, "client%d-text-only.jsonl" % client_id)
            with open(items_img_text_path, mode="w", encoding="utf-8") as writer:
                for item in items_img_text:
                    writer.write(json.dumps(item, indent=None))
                    writer.write('\n')
            with open(items_img_only_path, mode="w", encoding="utf-8") as writer:
                for item in items_img_only:
                    writer.write(json.dumps(item, indent=None))
                    writer.write('\n')
            with open(items_text_only_path, mode="w", encoding="utf-8") as writer:
                for item in items_text_only:
                    writer.write(json.dumps(item, indent=None))
                    writer.write('\n')
            print(f"finish write client{client_id}'s missing modal items into json file")
            modal_statistic[client_id] = {
                "img-text": len(items_img_text), "img-only": len(items_img_only), "text-only": len(items_text_only),
                "both-missing": data_statistic[client_id] - len(items_img_text) - len(items_img_only) - len(items_text_only)
            }

        data_statistic_path = os.path.join(data_path, split_file_name, "statistic.jsonl")
        with open(data_statistic_path, mode="w", encoding="utf-8") as writer:
            for client_id in range(n_clients):
                writer.write("client%d:" % int(client_id))
                writer.write(str(data_statistic[client_id]))
                writer.write(" img-text:%d" % modal_statistic[client_id]["img-text"])
                writer.write(" img-only:%d" % modal_statistic[client_id]["img-only"])
                writer.write(" text-only:%d" % modal_statistic[client_id]["text-only"])
                writer.write(" both-missing:%d" % modal_statistic[client_id]["both-missing"])
                writer.write('\n')
        print("finish modal-missing-non-iid data statistic")


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def create_dataset_by_split(args, split, is_train=True, is_no_grad=False):
    transform = build_transform(is_train=is_train, args=args)
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    dataset = VQAv2Dataset(
        data_path=args.data_path, split=split,
        transform=transform, tokenizer=tokenizer,
        num_max_bpe_tokens=args.num_max_bpe_tokens,
        task=args.task, **opt_kwargs,
    )
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    if is_no_grad:
        batch_size = args.eval_batch_size

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )


def iid_data_loader(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        create_dataset_by_split(args, split="train", is_train=True), \
        create_dataset_by_split(args, split="val", is_train=True)


if __name__ == '__main__':
    pass
    # run in Terminal
    from datasets.dataset import VQAv2Dataset
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer("init_weight/beit3.spm")
    VQAv2Dataset.make_dataset_index(data_path="./data", tokenizer=tokenizer, annotation_data_path="data/vqa", scale=False)
    VQAv2Dataset.make_modal_missing_index('./data', modal_missing_rate=0.5, alpha=0.1, n_clients=30, n_classes=310)



