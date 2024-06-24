import os
import glob
import torch
import copy
import torch.distributed as dist
from optims.optim import create_optimizer, LayerDecayValueAssigner, get_is_head_flag_for_vit
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from algorithm.BaseTrainer import train_one_epoch, VQAHandler, evaluate
from utils import utils
from pathlib import Path
from datasets.dataset import create_dataloader


class ClientTrainer:
    def __init__(self, args, model, client_id, server_path, client_path, steps_per_epoch, logger):
        self.args = args
        self.model = model
        self.client_id = client_id
        self.client_path = client_path
        self.server_path = server_path
        self.logger = logger
        self.num_training_steps_per_epoch = steps_per_epoch

        self.total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        self.assigner = LayerDecayValueAssigner([1.0, 20.0], scale_handler=get_is_head_flag_for_vit)

        self.lr_schedule_values = utils.cosine_scheduler(
            base_value=self.args.lr, final_value=self.args.min_lr, epochs=self.args.local_epochs,
            niter_per_ep=steps_per_epoch, warmup_epochs=self.args.warmup_epochs, sched_type="linear"
        )

        self.model_ema = None
        self.update_freq = 1
        self.clip_grad = None

        self.prototypes = {"img": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                           "text": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                           "fusion": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                           "target": torch.randn([args.num_classes, args.num_classes], dtype=torch.float32)}

    def run(self, cur_global_epoch, cur_local_epoch, step_offset, train_dataloader, val_dataloader, device, mode):

        if cur_local_epoch == 0 and step_offset == 0:
            self.load_model()

        self.model.to(device)
        for key in self.prototypes.keys():
            self.prototypes[key] = self.prototypes[key].to(device)

        loss_scaler = NativeScaler()

        model_ddp = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                              find_unused_parameters=True)

        optimizer = create_optimizer(
            self.args, self.model, skip_list=self.model.no_weight_decay(),
            get_num_layer=self.assigner.get_layer_id, get_layer_scale=self.assigner.get_scale
        )

        task_handler = VQAHandler()

        train_dataloader.sampler.set_epoch(cur_global_epoch)
        train_one_epoch(
            self.args, model_ddp, train_dataloader, optimizer, device, task_handler, cur_global_epoch,
            cur_local_epoch, cur_local_epoch * self.num_training_steps_per_epoch + step_offset,
            self.lr_schedule_values, loss_scaler, self.prototypes, self.clip_grad, self.update_freq,
            self.model_ema, mode, self.logger
        )
        torch.distributed.barrier()

        self.model.set_mode("vl")
        self.model.cpu()
        for key in self.prototypes.keys():
            self.prototypes[key] = self.prototypes[key].cpu()
        torch.distributed.barrier()

    def load_model(self):
        all_global_model = glob.glob(os.path.join(self.server_path, 'global_model-*.pth'))
        latest_ckpt = -1
        for ckpt in all_global_model:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            latest_global_model = os.path.join(self.server_path, 'global_model-%d.pth' % latest_ckpt)
            self.logger.write("load global model global_model-%d.pth" % latest_ckpt)
            self.model.load_state_dict(torch.load(latest_global_model))
        else:
            self.logger.write("communication round 0, start with pretrain params")

    def save_model(self, ckpt_name):
        if utils.is_main_process():
            state_dict = self.model.state_dict()
            torch.save(state_dict, self.client_path + "/" + ckpt_name)

    def compute_prototypes(self, dataloader, device):

        self.model.set_mode("vl")
        self.model.eval()
        self.model.to(device)

        img_rep_box = None
        text_rep_box = None
        fusion_rep_box = None
        target_box = None
        label_box = None

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                image = data["image"].to(device, non_blocking=True)
                language_tokens = data["language_tokens"].to(device, non_blocking=True)
                padding_mask = data["padding_mask"].to(device, non_blocking=True)
                logits, hidden_reps = self.model(image=image, question=language_tokens, padding_mask=padding_mask)

                img_rep = hidden_reps["img_rep"].clone().detach()
                text_rep = hidden_reps["text_rep"].clone().detach()
                fusion_rep = hidden_reps["fusion_rep"].clone().detach()
                soft_logit = logits.clone().detach() / self.args.kd_temperature
                target = torch.nn.functional.softmax(soft_logit, dim=-1)
                labels = data["labels"].to(device)

                img_rep_box = img_rep if img_rep_box is None else torch.cat([img_rep_box, img_rep], dim=0)
                text_rep_box = text_rep if text_rep_box is None else torch.cat([text_rep_box, text_rep], dim=0)
                fusion_rep_box = fusion_rep if fusion_rep_box is None else torch.cat([fusion_rep_box, fusion_rep], dim=0)
                target_box = target if target_box is None else torch.cat([target_box, target], dim=0)
                label_box = labels if label_box is None else torch.cat([label_box, labels], dim=0)

            img_prototype_sum = torch.matmul(label_box.T, img_rep_box)
            text_prototype_sum = torch.matmul(label_box.T, text_rep_box)
            fusion_prototype_sum = torch.matmul(label_box.T, fusion_rep_box)
            target_prototype_sum = torch.matmul(label_box.T, target_box)
            total_weight_per_class = torch.sum(label_box.T, dim=1, keepdim=True)

            world_size = utils.get_world_size()
            img_prototypes_sum_list = [torch.zeros_like(img_prototype_sum) for _ in range(world_size)]
            text_prototypes_sum_list = [torch.zeros_like(text_prototype_sum) for _ in range(world_size)]
            fusion_prototypes_sum_list = [torch.zeros_like(fusion_prototype_sum) for _ in range(world_size)]
            target_prototype_sum_list = [torch.zeros_like(target_prototype_sum) for _ in range(world_size)]
            total_weight_per_class_list = [torch.zeros_like(total_weight_per_class) for _ in range(world_size)]
            dist.all_gather(img_prototypes_sum_list, img_prototype_sum)
            dist.all_gather(text_prototypes_sum_list, text_prototype_sum)
            dist.all_gather(fusion_prototypes_sum_list, fusion_prototype_sum)
            dist.all_gather(target_prototype_sum_list, target_prototype_sum)
            dist.all_gather(total_weight_per_class_list, total_weight_per_class)

            for i in range(world_size):
                if i == 0:
                    all_total_weight_per_class = total_weight_per_class_list[i]
                    all_img_prototype_sum = img_prototypes_sum_list[i]
                    all_text_prototype_sum = text_prototypes_sum_list[i]
                    all_fusion_prototype_sum = fusion_prototypes_sum_list[i]
                    all_target_prototype_sum = target_prototype_sum_list[i]
                else:
                    all_total_weight_per_class += total_weight_per_class_list[i]
                    all_img_prototype_sum += img_prototypes_sum_list[i]
                    all_text_prototype_sum += text_prototypes_sum_list[i]
                    all_fusion_prototype_sum += fusion_prototypes_sum_list[i]
                    all_target_prototype_sum += target_prototype_sum_list[i]

            img_prototypes = all_img_prototype_sum / all_total_weight_per_class
            text_prototypes = all_text_prototype_sum / all_total_weight_per_class
            fusion_prototypes = all_fusion_prototype_sum / all_total_weight_per_class
            target_prototypes = all_target_prototype_sum / all_total_weight_per_class

            # replace Nan
            img_prototypes = torch.where(torch.isnan(img_prototypes), torch.full_like(img_prototypes, 0.0), img_prototypes)
            text_prototypes = torch.where(torch.isnan(text_prototypes), torch.full_like(text_prototypes, 0.0), text_prototypes)
            fusion_prototypes = torch.where(torch.isnan(fusion_prototypes), torch.full_like(fusion_prototypes, 0.0), fusion_prototypes)
            target_prototypes = torch.where(torch.isnan(target_prototypes), torch.full_like(target_prototypes, 0.0), target_prototypes)

            img_prototype_new = img_prototypes.detach().cpu()
            text_prototype_new = text_prototypes.detach().cpu()
            fusion_prototype_new = fusion_prototypes.detach().cpu()
            target_prototypes_new = target_prototypes.detach().cpu()

            zero_prototype = torch.zeros([1, self.args.embed_dim], device="cpu", dtype=torch.float32)
            for clas in range(self.args.num_classes):
                if not torch.equal(img_prototype_new[clas:clas+1, :], zero_prototype):
                    self.prototypes["img"][clas:clas+1, :] = img_prototype_new[clas:clas+1, :]
                if not torch.equal(text_prototype_new[clas:clas+1, :], zero_prototype):
                    self.prototypes["text"][clas:clas+1, :] = text_prototype_new[clas:clas+1, :]
                if not torch.equal(fusion_prototype_new[clas:clas+1, :], zero_prototype):
                    self.prototypes["fusion"][clas:clas+1, :] = fusion_prototype_new[clas:clas+1, :]
            zero_target = torch.zeros([1, self.args.num_classes], device="cpu", dtype=torch.float32)
            for clas in range(self.args.num_classes):
                if not torch.equal(target_prototypes_new[clas:clas+1, :], zero_target):
                    self.prototypes["target"][clas:clas+1, :] = target_prototypes_new[clas:clas+1, :]

        self.model.cpu()
        self.model.train()
        self.logger.write("finish computing prototypes.")

    def save_prototypes(self, ckpt_name):
        if utils.is_main_process():
            torch.save(self.prototypes, self.client_path + "/" + ckpt_name)

    def load_global_prototypes(self):
        all_global_proto = glob.glob(os.path.join(self.server_path, 'global_prototypes-*.pth'))
        latest_proto = -1
        for proto in all_global_proto:
            t = proto.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_proto = max(int(t), latest_proto)
        if latest_proto >= 0:
            global_prototypes_path = os.path.join(self.server_path, 'global_prototypes-%d.pth' % latest_proto)
            self.logger.write("load global prototypes global_prototypes-%d.pth" % latest_proto)
            self.prototypes = torch.load(global_prototypes_path)
        else:
            self.logger.write("no global prototypes!")


