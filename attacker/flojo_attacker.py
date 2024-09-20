import gc
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, vprint, get_target_hr, goal_oriented_loss, AttackDataset, gumbel_topk
import torch.nn.functional as F
import time
import os
import scipy.sparse as sp
from torch.optim import SGD, Adam
import higher


class FLOJOAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(FLOJOAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.tau = attacker_config.get('tau', 1.)
        self.expected_hr = attacker_config['expected_hr']
        self.step = attacker_config['step']
        self.n_adv_epochs = attacker_config['n_adv_epochs']
        self.n_retraining_epochs = attacker_config['n_retraining_epochs']
        self.lr = attacker_config['lr']
        self.reg = attacker_config['reg']
        self.momentum = attacker_config['momentum']
        self.look_ahead_step = attacker_config['look_ahead_step']
        self.look_ahead_lr = attacker_config['look_ahead_lr']

        self.candidate_mat = self.construct_candidates()
        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=False)

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
            top_scores.append(scores.topk(self.topk, dim=1).values[:, -1:])
        target_scores = torch.cat(target_scores, dim=0)
        top_scores = torch.cat(top_scores, dim=0)
        return target_scores, top_scores

    def fake_train(self, surrogate_trainer, fake_tensor):
        with torch.no_grad():
            profiles = gumbel_topk(fake_tensor, self.n_inters, self.tau)
        attack_dataset = AttackDataset(profiles, self.dataset.n_users,
                                       surrogate_trainer.negative_sample_ratio)
        attack_dataloader = DataLoader(attack_dataset, batch_size=surrogate_trainer.dataloader.batch_size,
                                       num_workers=surrogate_trainer.dataloader.num_workers,
                                       persistent_workers=False, pin_memory=False)
        original_dataloader = surrogate_trainer.dataloader
        surrogate_trainer.dataloader = attack_dataloader
        tf_loss = surrogate_trainer.train_one_epoch(None)
        surrogate_trainer.dataloader = original_dataloader
        return tf_loss

    def train_fake(self, surrogate_model, surrogate_trainer, fake_tensor, adv_opt,
                   temp_fake_user_tensor, verbose):
        opt = SGD(surrogate_model.parameters(), lr=self.look_ahead_lr)
        with higher.innerloop_ctx(surrogate_model, opt) as (fmodel, diffopt):
            fmodel.train()
            for s in range(self.look_ahead_step):
                profiles = gumbel_topk(fake_tensor, self.n_inters, self.tau, hard=False)
                n_samples = profiles.sum()
                n_valid_fakes = torch.gt(profiles, 0).any(dim=1).float().sum()
                weight_per_fake = n_samples / n_valid_fakes

                profiles = profiles - torch.maximum(profiles.detach() - 1, torch.zeros_like(profiles))
                n_profiles = 1. - profiles
                profiles = F.normalize(profiles, p=1, dim=1) * weight_per_fake
                n_profiles = F.normalize(n_profiles, p=1,
                                         dim=1) * weight_per_fake * surrogate_trainer.negative_sample_ratio

                scores, l2_norm_sq = fmodel.forward(temp_fake_user_tensor)
                loss_p = F.softplus(-scores) + l2_norm_sq * surrogate_trainer.l2_reg
                loss_n = F.softplus(scores) + l2_norm_sq * surrogate_trainer.l2_reg
                loss_p = (loss_p * profiles).sum()
                loss_n = (loss_n * n_profiles).sum()
                loss_fake = loss_p + loss_n

                loss_normal = 0.
                for batch_data in surrogate_trainer.dataloader:
                    inputs = batch_data.to(device=self.device, dtype=torch.int64)
                    pos_users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
                    inputs = inputs.reshape(-1, 3)
                    neg_users, neg_items = inputs[:, 0], inputs[:, 2]
                    pos_scores, neg_scores, l2_norm_sq = fmodel.bce_forward(pos_users, pos_items, neg_users, neg_items)
                    loss_p = F.softplus(-pos_scores)
                    loss_n = F.softplus(neg_scores)
                    loss_normal += loss_p.sum() + loss_n.sum() + l2_norm_sq.sum() * surrogate_trainer.l2_reg
                diffopt.step(loss_fake + loss_normal)
                vprint('Unroll step {:d}, Train Loss: {:.6f}'.format(s, loss_fake + loss_normal), verbose)

            fmodel.eval()
            target_scores, top_scores = self.get_target_item_and_top_scores(fmodel)
            adv_loss = goal_oriented_loss(target_scores, top_scores, self.expected_hr).sum()
            adv_grads = torch.autograd.grad(adv_loss, fake_tensor)[0]

        adv_opt.zero_grad()
        fake_tensor.grad = adv_grads
        adv_opt.step()
        with torch.no_grad():
            _, items = fake_tensor.topk(self.n_inters, dim=1)
            fake_tensor.data -= self.reg
            fake_tensor.data += torch.zeros_like(fake_tensor).scatter(1, items, 2 * self.reg)
        vprint('Adversarial Loss: {:.6f}'.format(adv_loss.item()), verbose)
        return adv_loss.item()

    def init_fake_tensor(self, n_temp_fakes):
        sample_idxes = torch.randint(self.candidate_mat.shape[0], size=[n_temp_fakes])
        fake_tensor = torch.tensor(self.candidate_mat[sample_idxes].toarray() * 5., dtype=torch.float32, device=self.device)
        fake_tensor.requires_grad = True
        return fake_tensor

    def choose_filler_items(self, fake_tensor, temp_fake_user_tensor):
        with torch.no_grad():
            _, items = fake_tensor.topk(self.n_inters, dim=1)
            n_inters = torch.gt(fake_tensor, 0.).int().sum(dim=1)
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            filler_items = items[u_idx, :min(n_inters[u_idx], self.n_inters)]

            f_u = temp_fake_user_tensor[u_idx]
            filler_items = filler_items.cpu().numpy().tolist()
            self.fake_user_inters[f_u - self.n_users] = filler_items
            self.dataset.train_data.append(set(filler_items))
            self.dataset.val_data.append({})
            self.dataset.train_array += [[f_u, item] for item in filler_items]
            self.dataset.n_users += 1

    def retrain_surrogate(self, fake_tensor, adv_opt, temp_fake_user_tensor, fake_nums_str,
                          adv_epoch, verbose, writer):
        surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)
        for retraining_epoch in range(self.n_retraining_epochs):
            start_time = time.time()

            surrogate_model.train()
            adv_loss = self.train_fake(surrogate_model, surrogate_trainer, fake_tensor, adv_opt,
                                       temp_fake_user_tensor, verbose)
            tn_loss = surrogate_trainer.train_one_epoch(None)
            tf_loss = self.fake_train(surrogate_trainer, fake_tensor)
            if retraining_epoch == self.n_retraining_epochs:
                self.train_fake(surrogate_model, surrogate_trainer, fake_tensor, adv_opt, temp_fake_user_tensor,
                                verbose)

            target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)
            consumed_time = time.time() - start_time
            vprint('Adversarial Epoch {:d}/{:d}, Retraining Epoch {:d}/{:d}, Time: {:.3f}s, '
                   'Train Loss Normal: {:.6f}, Train Loss Fake: {:.6f}, '
                   'Adversarial Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                   format(adv_epoch, self.n_adv_epochs, retraining_epoch, self.n_retraining_epochs, consumed_time,
                          tn_loss, tf_loss, adv_loss, target_hr * 100.), verbose)
            global_retraining_epoch = adv_epoch * self.n_retraining_epochs + retraining_epoch
            writer_tag = '{:s}_{:s}'.format(self.name, fake_nums_str)
            if writer:
                writer.add_scalar(writer_tag + '/Train_Loss_Normal', tn_loss, global_retraining_epoch)
                writer.add_scalar(writer_tag + '/Train_Loss_Fake', tf_loss, global_retraining_epoch)
                writer.add_scalar(writer_tag + '/Adv_Loss', adv_loss, global_retraining_epoch)
                writer.add_scalar(writer_tag + '/Hit_Ratio@' + str(self.topk), target_hr, global_retraining_epoch)

    def generate_fake_users(self, verbose=True, writer=None):
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            start_time = time.time()
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating fake #{:s} !'.format(fake_nums_str))
            temp_fake_user_tensor = np.arange(fake_user_end_indices[i_step - 1],
                                              fake_user_end_indices[i_step]) + self.n_users
            temp_fake_user_tensor = torch.tensor(temp_fake_user_tensor, dtype=torch.int64, device=self.device)
            n_temp_fakes = temp_fake_user_tensor.shape[0]

            self.surrogate_model_config['n_fakes'] = n_temp_fakes
            fake_tensor = self.init_fake_tensor(n_temp_fakes)
            adv_opt = SGD([fake_tensor], lr=self.lr)

            for adv_epoch in range(self.n_adv_epochs):
                self.retrain_surrogate(fake_tensor, adv_opt, temp_fake_user_tensor, fake_nums_str,
                                       adv_epoch, verbose, writer)

            self.choose_filler_items(fake_tensor, temp_fake_user_tensor)
            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            print('Fake #{:s} has been generated! Time: {:.3f}s'.format(fake_nums_str, consumed_time))
            gc.collect()
            torch.cuda.empty_cache()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
