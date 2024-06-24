# Multi Model Federated Learning algorithm
from timm.models import create_model
from algorithm.ClientTrainer import ClientTrainer
from utils import utils
from pathlib import Path
import random
import os
import glob
import torch
import numpy as np
from datasets.dataset import create_dataset_by_split


class MMFL:
    def __init__(self, args, client_nums, sample_num_dict, device, val_split, train_split_list_img_text,
                 train_split_list_img_only, train_split_list_text_only, logger):
        self.args = args
        self.client_nums = client_nums
        self.sample_num_dict = sample_num_dict
        self.device = device
        self.val_split = val_split
        self.train_split_list_img_text = train_split_list_img_text
        self.train_split_list_img_only = train_split_list_img_only
        self.train_split_list_text_only = train_split_list_text_only
        self.logger = logger

        self.client_path = None
        self.server_path = args.output_dir + "/server"
        Path(self.server_path).mkdir(parents=True, exist_ok=True)

        self.model_config = "beit3_base_patch16_480_vqav2"

    def run(self, n_comm_rounds, selected_client_nums):

        party_nums = max(1, selected_client_nums)

        for cur_round in range(1, n_comm_rounds + 1):
            # set mse loss
            if self.args.prototype_as_rep_target:
                if cur_round >= self.args.regularization_start_round:
                    if not self.args.no_img_proto_target:
                        self.args.img_proto_target = True
                    if not self.args.no_text_proto_target:
                        self.args.text_proto_target = True
                    if not self.args.no_fusion_proto_target:
                        self.args.fusion_proto_target = True

            # select clients
            self.logger.write("in communication round %d." % cur_round)
            self.client_path = self.args.output_dir + '/client-round%d' % cur_round
            Path(self.client_path).mkdir(parents=True, exist_ok=True)

            party_list = random.sample([i for i in range(self.client_nums)], party_nums)
            self.logger.write("party list:")
            self.logger.write(party_list)

            # local train
            start_global_epoch = (cur_round - 1) * self.args.local_epochs
            num_epochs = self.args.local_epochs
            for client_id in party_list:
                self.logger.write("----- start to train client%d in round%d ------" % (client_id, cur_round))

                model = create_model(
                    self.model_config,
                    pretrained=False,
                    drop_path_rate=self.args.drop_path,
                    vocab_size=self.args.vocab_size,
                    checkpoint_activations=self.args.checkpoint_activations,
                )
                utils.load_model_and_may_interpolate(
                    ckpt_path="./init_weight/beit3_base_patch16_224.pth",
                    model=model, model_key='model|module', model_prefix=''
                )
                self.logger.write("finish create model.")

                global_batch_size = self.args.batch_size * utils.get_world_size()
                steps_per_epoch = self.sample_num_dict[client_id]["total"] // global_batch_size
                self.logger.write("steps_per_epoch is not more than %d" % steps_per_epoch)

                client_trainer = ClientTrainer(self.args, model=model, client_id=client_id,
                                               client_path=self.client_path, server_path=self.server_path,
                                               steps_per_epoch=steps_per_epoch, logger=self.logger)
                self.logger.write("finish creating client trainer.")

                if self.args.prototype_as_rep_target or self.args.prototype_as_missing_modal:
                    client_trainer.load_global_prototypes()
                    self.logger.write("finish loading global prototypes.")

                # get data loader
                train_loader_img_text = create_dataset_by_split(self.args,
                                                                split=self.train_split_list_img_text[client_id],
                                                                is_train=True)
                val_loader = create_dataset_by_split(self.args, split=self.val_split, is_train=True)
                loader_for_prototype = create_dataset_by_split(self.args,
                                                               split=self.train_split_list_img_text[client_id],
                                                               is_train=True, is_no_grad=True)  # randaug is used
                if self.args.modal_missing:
                    train_loader_img_only = create_dataset_by_split(self.args,
                                                                    split=self.train_split_list_img_only[client_id],
                                                                    is_train=True)
                    train_loader_text_only = create_dataset_by_split(self.args,
                                                                     split=self.train_split_list_text_only[client_id],
                                                                     is_train=True)
                self.logger.write("finish create all dataloader.")
                torch.distributed.barrier()

                # run
                self.logger.write("start training client:")
                for cur_local_epoch in range(num_epochs):
                    self.logger.write("start training client img-text.")
                    if self.sample_num_dict[client_id]["img-text"] >= global_batch_size:
                        client_trainer.run(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                           cur_local_epoch=cur_local_epoch, step_offset=0,
                                           train_dataloader=train_loader_img_text, val_dataloader=val_loader,
                                           device=self.device, mode="vl")
                        torch.distributed.barrier()

                    if self.args.modal_missing:
                        if np.random.binomial(n=1, p=0.5, size=1)[0]:
                            self.logger.write("train img-only first.")
                            offset_step = self.sample_num_dict[client_id]["img-text"] // global_batch_size
                            self.logger.write("offset step is %d" % offset_step)
                            if self.sample_num_dict[client_id]["img-only"] >= global_batch_size:
                                client_trainer.run(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                   cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                   train_dataloader=train_loader_img_only, val_dataloader=val_loader,
                                                   device=self.device, mode="v")
                                torch.distributed.barrier()
                            offset_step += self.sample_num_dict[client_id]["img-only"] // global_batch_size
                            self.logger.write("offset step is %d" % offset_step)
                            if self.sample_num_dict[client_id]["text-only"] >= global_batch_size:
                                client_trainer.run(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                   cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                   train_dataloader=train_loader_text_only, val_dataloader=val_loader,
                                                   device=self.device, mode="l")
                        else:
                            self.logger.write("train text-only first.")
                            offset_step = self.sample_num_dict[client_id]["img-text"] // global_batch_size
                            self.logger.write("offset step is %d" % offset_step)
                            if self.sample_num_dict[client_id]["text-only"] >= global_batch_size:
                                client_trainer.run(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                   cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                   train_dataloader=train_loader_text_only, val_dataloader=val_loader,
                                                   device=self.device, mode="l")
                                torch.distributed.barrier()
                            offset_step += self.sample_num_dict[client_id]["text-only"] // global_batch_size
                            self.logger.write("offset step is %d" % offset_step)
                            if self.sample_num_dict[client_id]["img-only"] >= global_batch_size:
                                client_trainer.run(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                   cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                   train_dataloader=train_loader_img_only, val_dataloader=val_loader,
                                                   device=self.device, mode="v")
                        torch.distributed.barrier()

                self.logger.write("finish training client%d in round%d " % (client_id, cur_round))
                torch.distributed.barrier()
                client_trainer.save_model("client%d_model.pth" % client_id)

                # get prototype
                if self.args.prototype_as_rep_target or self.args.prototype_as_missing_modal:
                    client_trainer.compute_prototypes(dataloader=loader_for_prototype, device=self.device)
                    client_trainer.save_prototypes(ckpt_name="client%d_prototypes.pth" % client_id)
                    self.logger.write("get prototypes for client%d in round%d " % (client_id, cur_round))
                    torch.distributed.barrier()

            self.logger.write("---------- finish training in round%d ----------" % cur_round)

            # avg
            self.logger.write("start fedavg.")
            self.fedavg(party_list=party_list, cur_round=cur_round)
            self.logger.write("finish fedavg.")
            torch.distributed.barrier()

            # aggregate
            if self.args.prototype_as_rep_target or self.args.prototype_as_missing_modal:
                self.logger.write("start aggregating global prototype.")
                self.aggregate_local_prototypes(party_list=party_list, cur_round=cur_round)
                self.logger.write("finish aggregating global prototype.")
                torch.distributed.barrier()

            print("--------------------- finish round%d ----------------------" % cur_round)

        print("finish MMFL.")

    def fedavg(self, party_list, cur_round, aggregation_factor="total"):
        if utils.is_main_process():
            global_state_dict = {}

            # get fedavg weight for each client
            sample_per_party_client = []
            for client_id in party_list:
                sample_per_party_client.append(float(self.sample_num_dict[client_id][aggregation_factor]))
            total_sample_cur_round = sum(sample_per_party_client)
            fed_avg_freqs = [i / total_sample_cur_round for i in sample_per_party_client]
            self.logger.write("fedavg weight:")
            self.logger.write(fed_avg_freqs)

            # get parameters for each client
            state_dicts = []
            all_client_model_path = []
            for client_id in party_list:
                all_client_model_path.append(os.path.join(self.client_path, 'client%d_model.pth' % client_id))
            for ckpt in all_client_model_path:
                state_dicts.append(torch.load(ckpt))
            for idx, state_dict in enumerate(state_dicts):
                if idx == 0:
                    for key in state_dict:
                        global_state_dict[key] = state_dict[key] * fed_avg_freqs[idx]
                else:
                    for key in state_dict:
                        global_state_dict[key] += state_dict[key] * fed_avg_freqs[idx]

            # save global model
            torch.save(global_state_dict, self.server_path + '/' + 'global_model-%d.pth' % cur_round)
            print("finishe saving global model.")

    def aggregate_local_prototypes(self, party_list, cur_round):
        if utils.is_main_process():

            global_prototypes = {}

            # get aggregation weight for prototypes
            img_sample_per_party_client = []
            text_sample_per_party_client = []
            fusion_sample_per_party_client = []
            target_sample_per_party_client = []
            for client_id in party_list:
                img_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"] + self.sample_num_dict[client_id]["img-only"]
                text_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"] + self.sample_num_dict[client_id]["text-only"]
                fusion_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"]
                target_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"]
                img_sample_per_party_client.append(float(img_sample_nums))
                text_sample_per_party_client.append(float(text_sample_nums))
                fusion_sample_per_party_client.append(float(fusion_sample_nums))
                target_sample_per_party_client.append(float(target_sample_nums))
            img_proto_aggre_weight = [i / sum(img_sample_per_party_client) for i in img_sample_per_party_client]
            text_proto_aggre_weight = [i / sum(text_sample_per_party_client) for i in text_sample_per_party_client]
            fusion_proto_aggre_weight = [i / sum(fusion_sample_per_party_client) for i in fusion_sample_per_party_client]
            target_proto_aggre_weight = [i / sum(target_sample_per_party_client) for i in target_sample_per_party_client]

            # get clients' prototypes
            all_client_prototypes_box = []
            all_client_prototype_path = []
            for client_id in party_list:
                all_client_prototype_path.append(os.path.join(self.client_path, 'client%d_prototypes.pth' % client_id))
            for prototype_path in all_client_prototype_path:
                all_client_prototypes_box.append(torch.load(prototype_path))
            for idx, prototypes in enumerate(all_client_prototypes_box):
                if idx == 0:
                    global_prototypes["img"] = prototypes["img"] * img_proto_aggre_weight[idx]
                    global_prototypes["text"] = prototypes["text"] * text_proto_aggre_weight[idx]
                    global_prototypes["fusion"] = prototypes["fusion"] * fusion_proto_aggre_weight[idx]
                    global_prototypes["target"] = prototypes["target"] * target_proto_aggre_weight[idx]
                else:
                    global_prototypes["img"] += prototypes["img"] * img_proto_aggre_weight[idx]
                    global_prototypes["text"] += prototypes["text"] * text_proto_aggre_weight[idx]
                    global_prototypes["fusion"] += prototypes["fusion"] * fusion_proto_aggre_weight[idx]
                    global_prototypes["target"] += prototypes["target"] * target_proto_aggre_weight[idx]

            # regular (not affect by zero-prototype)
            zero_proto = torch.zeros([1, self.args.embed_dim], device="cpu", dtype=torch.float32)
            zero_target = torch.zeros([1, self.args.num_classes], device="cpu", dtype=torch.float32)
            total_img_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            total_text_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            total_fusion_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            total_target_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            for idx, prototypes in enumerate(all_client_prototypes_box):
                for clas in range(self.args.num_classes):
                    if not torch.equal(prototypes["img"][clas:clas+1, :], zero_proto):
                        total_img_weight_per_class[clas] += img_proto_aggre_weight[idx]
                    if not torch.equal(prototypes["text"][clas:clas+1, :], zero_proto):
                        total_text_weight_per_class[clas] += text_proto_aggre_weight[idx]
                    if not torch.equal(prototypes["fusion"][clas:clas+1, :], zero_proto):
                        total_fusion_weight_per_class[clas] += fusion_proto_aggre_weight[idx]
                    if not torch.equal(prototypes["target"][clas:clas+1, :], zero_target):
                        total_target_weight_per_class[clas] += target_proto_aggre_weight[idx]
            for clas in range(self.args.num_classes):
                if np.abs(total_img_weight_per_class[clas]) > 0.0001:
                    global_prototypes["img"][clas:clas+1, :] /= total_img_weight_per_class[clas]
                if np.abs(total_text_weight_per_class[clas]) > 0.0001:
                    global_prototypes["text"][clas:clas+1, :] /= total_text_weight_per_class[clas]
                if np.abs(total_fusion_weight_per_class[clas]) > 0.0001:
                    global_prototypes["fusion"][clas:clas+1, :] /= total_fusion_weight_per_class[clas]
                if np.abs(total_target_weight_per_class[clas]) > 0.0001:
                    global_prototypes["target"][clas:clas+1, :] /= total_target_weight_per_class[clas]

            # save global prototypes
            torch.save(global_prototypes, os.path.join(self.server_path, 'global_prototypes-%d.pth' % cur_round))


