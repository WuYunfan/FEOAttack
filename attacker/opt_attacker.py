import gc
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, bce_loss, vprint, get_target_hr, dada_loss, gumbel_topk
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

        self.b1 = attacker_config.get('b1', 1.)
        self.b2 = attacker_config.get('b2', 1.)
        self.lmd1 = attacker_config['lmd1']
        self.lmd2 = attacker_config['lmd2']
        self.tau = attacker_config.get('tau', 1.)
        self.step = attacker_config['step']
        self.n_rounds = attacker_config['n_rounds']
        self.n_fake_epochs = attacker_config['n_fake_epochs']
        self.lr = attacker_config['lr']
        self.weight_decay = attacker_config['weight_decay']

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

    def get_target_item_scores(self, surrogate_model):
        scores = []
        for users in self.target_user_loader:
            users = users[0]
            scores.append(surrogate_model.predict(users)[:, self.target_item_tensor])
        scores = torch.cat(scores, dim=0)
        return scores

    def fake_train(self, surrogate_model, surrogate_trainer, fake_tensor, adv_opt, temp_fake_user_tensor):
        fake_tensor_topk = gumbel_topk(fake_tensor, self.n_inters, self.tau)
        scores, l2_norm_sq = surrogate_model.forward(temp_fake_user_tensor)
        b_loss = bce_loss(fake_tensor_topk, scores, surrogate_trainer.negative_sample_ratio)
        reg_loss = surrogate_trainer.l2_reg * l2_norm_sq.mean()
        loss = b_loss + reg_loss
        surrogate_trainer.opt.zero_grad()
        adv_opt.zero_grad()
        loss.backward()
        surrogate_trainer.opt.step()
        adv_opt.step()
        return loss.item()

    def poison_train(self, surrogate_model, surrogate_trainer, pre_scores):
        scores = self.get_target_item_scores(surrogate_model)
        loss = dada_loss(scores, pre_scores, self.b1, self.b2, self.lmd1, self.lmd2).mean()
        surrogate_trainer.opt.zero_grad()
        loss.backward()
        surrogate_trainer.opt.step()
        return loss.item()

    def init_fake_tensor(self, n_temp_fakes):
        sample_idxes = torch.randint(self.candidate_mat.shape[0], size=[n_temp_fakes])
        fake_tensor = torch.tensor(self.candidate_mat[sample_idxes].toarray(), dtype=torch.float32, device=self.device)
        fake_tensor.requires_grad = True
        return fake_tensor

    def choose_filler_items(self, fake_tensor, temp_fake_user_tensor):
        with torch.no_grad():
            fake_tensor_topk = gumbel_topk(fake_tensor, self.n_inters, self.tau)
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            filler_items = fake_tensor_topk[u_idx, :].topk(self.n_inters).indices

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
                with torch.no_grad():
                    pre_scores = self.get_target_item_scores(surrogate_model)
                for f_epcoh in range(self.n_fake_epochs):
                    tf_loss = self.fake_train(surrogate_model, surrogate_trainer, fake_tensor, adv_opt, temp_fake_user_tensor)
                    vprint('Fake Epoch {:d}/{:d}, Train Loss Fake: {:.6f}'.
                           format(f_epcoh, self.n_fake_epochs, tf_loss), verbose)
                p_loss = self.poison_train(surrogate_model, surrogate_trainer, pre_scores)

                target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)
                vprint('Round {:d}/{:d}, Poison Loss: {:.6f}, Train Loss Normal: {:.6f}, '
                       'Train Loss Fake: {:.6f}, Target Hit Ratio {:.6f}%'.
                       format(i_round, self.n_rounds, p_loss, tn_loss, tf_loss, target_hr * 100.), verbose)
                if writer:
                    writer.add_scalar('{:s}_{:s}/Poison_Loss'.format(self.name, fake_nums_str), p_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Train_Loss_Normal'.format(self.name, fake_nums_str), tn_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Train_Loss_Fake'.format(self.name, fake_nums_str), tf_loss, i_round)
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
