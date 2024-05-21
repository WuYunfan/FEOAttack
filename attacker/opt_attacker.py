import gc
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, vprint, get_target_hr, opt_loss, AttackDataset, bce_loss
import torch.nn.functional as F
import time
import os
import scipy.sparse as sp
from torch.optim import Adam


class OptAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(OptAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.alpha = attacker_config['alpha']
        self.init_hr = attacker_config['init_hr']
        self.hr_gain = attacker_config['hr_gain']
        self.step = attacker_config['step']
        self.n_rounds = attacker_config['n_rounds']
        self.lr = attacker_config['lr']
        self.weight_decay = attacker_config['weight_decay']
        self.n_fake_epochs = attacker_config['n_fake_epochs']

        self.candidate_mat = self.construct_candidates()
        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=True)

    def construct_candidates(self):
        data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                 shape=(self.dataset.n_users, self.dataset.n_items), dtype=np.float32).tocsc()
        mask = np.zeros(data_mat.shape[0], dtype=bool)
        for item in self.target_items:
            start_idx = data_mat.indptr[item]
            end_idx = data_mat.indptr[item + 1]
            row_indices = data_mat.indices[start_idx:end_idx]
            mask[row_indices] = True
        candidate_mat = data_mat.tocsr()[mask]
        return candidate_mat

    def get_target_item_and_top_scores(self, surrogate_model):
        target_scores = []
        top_scores = []
        for users in self.target_user_loader:
            users = users[0]
            scores = surrogate_model.predict(users)
            target_scores.append(scores[:, self.target_item_tensor])
            top_scores.append(scores.topk(self.topk).values[:, -1:])
        target_scores = torch.cat(target_scores, dim=0)
        top_scores = torch.cat(top_scores, dim=0)
        return target_scores, top_scores

    def fake_train(self, surrogate_model, surrogate_trainer, fake_tensor):
        profiles = F.softmax(fake_tensor, dim=-1)
        n_profiles = 1. - profiles
        dataset = AttackDataset(profiles, n_profiles, fake_tensor.shape[0] * self.n_inters, surrogate_trainer.negative_sample_ratio)
        dataloader = DataLoader(dataset, batch_size=surrogate_trainer.dataloader.batch_size,
                                num_workers=surrogate_trainer.dataloader.num_workers,
                                persistent_workers=False, pin_memory=False)
        losses = AverageMeter()
        for batch_data in dataloader:
            inputs = batch_data.to(device=self.device, dtype=torch.int64)
            pos_users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
            inputs = inputs.reshape(-1, 3)
            neg_users, neg_items = inputs[:, 0], inputs[:, 2]
            pos_scores, neg_scores, l2_norm_sq = surrogate_model.bce_forward(pos_users, pos_items, neg_users, neg_items)
            bce_loss_p = F.softplus(-pos_scores) * profiles[pos_users, pos_items]
            bce_loss_n = F.softplus(neg_scores) * n_profiles[neg_users, neg_items]

            bce_loss = torch.cat([bce_loss_p, bce_loss_n], dim=0).mean()
            reg_loss = surrogate_trainer.l2_reg * l2_norm_sq.mean()
            loss = bce_loss + reg_loss
            surrogate_trainer.opt.zero_grad()
            loss.backward()
            surrogate_trainer.opt.step()
            losses.update(loss.item(), l2_norm_sq.shape[0])
        return losses.avg

    def poison_train(self, surrogate_model, surrogate_trainer, target_hr):
        target_scores, top_scores = self.get_target_item_and_top_scores(surrogate_model)
        loss = self.alpha * opt_loss(target_scores, top_scores, target_hr).mean()
        surrogate_trainer.opt.zero_grad()
        loss.backward()
        surrogate_trainer.opt.step()
        return loss.item()

    def train_fake(self, surrogate_model, surrogate_trainer, fake_tensor, adv_opt, temp_fake_user_tensor, verbose):
        for f_epoch in range(self.n_fake_epochs):
            profiles = F.softmax(fake_tensor, dim=-1)
            scores = surrogate_model.predict(temp_fake_user_tensor)
            loss = bce_loss(profiles, scores, surrogate_trainer.negative_sample_ratio)
            adv_opt.zero_grad()
            loss.backward()
            adv_opt.step()
            vprint('Fake Epoch {:d}/{:d}, Train Loss Fake: {:.6f}'.format(f_epoch, self.n_fake_epochs, loss), verbose)
        return loss.item()

    def init_fake_tensor(self, n_temp_fakes):
        sample_idxes = torch.randint(self.candidate_mat.shape[0], size=[n_temp_fakes])
        fake_tensor = torch.tensor(self.candidate_mat[sample_idxes].toarray(), dtype=torch.float32, device=self.device)
        fake_tensor.requires_grad = True
        return fake_tensor

    def choose_filler_items(self, fake_tensor, temp_fake_user_tensor):
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            filler_items = fake_tensor[u_idx, :].topk(self.n_inters).indices

            f_u = temp_fake_user_tensor[u_idx]
            filler_items = filler_items.cpu().numpy().tolist()
            self.fake_user_inters[f_u - self.n_users] = filler_items
            self.dataset.train_data.append(set(filler_items))
            self.dataset.val_data.append({})
            self.dataset.train_array += [[f_u, item] for item in filler_items]
            self.dataset.n_users += 1

    def generate_fake_users(self, verbose=True, writer=None):
        start_time = time.time()
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating poison #{:s} !'.format(fake_nums_str))
            temp_fake_user_tensor = np.arange(fake_user_end_indices[i_step - 1],
                                              fake_user_end_indices[i_step]) + self.n_users
            temp_fake_user_tensor = torch.tensor(temp_fake_user_tensor, dtype=torch.int64, device=self.device)
            n_temp_fakes = temp_fake_user_tensor.shape[0]

            self.surrogate_model_config['n_fakes'] = n_temp_fakes
            surrogate_model = get_model(self.surrogate_model_config, self.dataset)
            surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)
            fake_tensor = self.init_fake_tensor(n_temp_fakes)
            adv_opt = Adam([fake_tensor], lr=self.lr, weight_decay=self.weight_decay)

            for i_round in range(self.n_rounds):
                surrogate_model.train()
                tn_loss = surrogate_trainer.train_one_epoch(None)
                tf_loss_r = self.fake_train(surrogate_model, surrogate_trainer, fake_tensor)
                targe_hr = self.hr_gain * fake_user_end_indices[i_step] / self.n_fakes + self.init_hr
                p_loss = self.poison_train(surrogate_model, surrogate_trainer, targe_hr)
                tf_loss_f = self.train_fake(surrogate_model, surrogate_trainer,
                                            fake_tensor, adv_opt, temp_fake_user_tensor, verbose)

                target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)
                vprint('Round {:d}/{:d}, Poison Loss: {:.6f}, Train Loss Normal: {:.6f}, '
                       'Train Loss Fake: {:.6f}, Fake Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                       format(i_round, self.n_rounds, p_loss, tn_loss, tf_loss_r, tf_loss_f, target_hr * 100.), verbose)
                if writer:
                    writer.add_scalar('{:s}_{:s}/Poison_Loss'.format(self.name, fake_nums_str), p_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Train_Loss_Normal'.format(self.name, fake_nums_str), tn_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Train_Loss_Fake'.format(self.name, fake_nums_str), tf_loss_r, i_round)
                    writer.add_scalar('{:s}_{:s}/Fake_Loss'.format(self.name, fake_nums_str), tf_loss_f, i_round)
                    writer.add_scalar('{:s}_{:s}/Hit_Ratio@{:d}'.format(self.name, fake_nums_str, self.topk),
                                      target_hr, i_round)
            self.choose_filler_items(fake_tensor, temp_fake_user_tensor)
            print('Poison #{:s} has been generated!'.format(fake_nums_str))
            gc.collect()
            torch.cuda.empty_cache()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
        consumed_time = time.time() - start_time
        self.consumed_time += consumed_time
