import gc
import random
import torch
from torch.cuda import device
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, vprint, get_target_hr, goal_oriented_loss, AttackDataset, gumbel_topk, PartialDataLoader
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

        self.expected_hr = attacker_config['expected_hr']
        self.step = attacker_config['step']
        self.n_adv_epochs = attacker_config['n_adv_epochs']
        self.n_retraining_epochs = attacker_config['n_retraining_epochs']
        self.lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']
        self.consistency_reg = attacker_config['consistency_reg']
        self.entropy_reg = attacker_config['entropy_reg']
        self.l2_reg = attacker_config['l2_reg']
        self.look_ahead_lr = attacker_config['look_ahead_lr']
        self.look_ahead_step = attacker_config['look_ahead_step']
        self.train_fake_its = attacker_config['train_fake_its']
        self.top_rate = attacker_config['top_rate']

        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=False)
        self.candidate_items = self.construct_candidate_items()
        self.fake_inters = 0

    def construct_candidate_items(self):
        n_top_items = int(self.n_items * self.top_rate)
        data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_popularity = np.array(np.sum(data_mat, axis=0)).squeeze()
        popularity_rank = np.argsort(item_popularity)[::-1].copy()
        popular_items = popularity_rank[:n_top_items]
        popular_candidate_tensor = torch.tensor(list(set(popular_items) - set(self.target_items)),
                                                dtype=torch.int64, device=self.device)
        popular_candidate_tensor = torch.cat([popular_candidate_tensor, self.target_item_tensor], dim=0)
        return popular_candidate_tensor

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

    def train_rec(self, surrogate_trainer, fake_tensor):
        with torch.no_grad():
            profiles = gumbel_topk(fake_tensor, self.n_inters)
        attack_dataset = AttackDataset(profiles, self.candidate_items, self.n_items, self.dataset.n_users)
        poisoned_dataset = ConcatDataset([self.dataset, attack_dataset])
        poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=surrogate_trainer.dataloader.batch_size,
                                         num_workers=surrogate_trainer.dataloader.num_workers,
                                         persistent_workers=False, pin_memory=False)
        surrogate_trainer.dataloader = poisoned_dataloader
        t_loss = surrogate_trainer.train_one_epoch(None)
        return t_loss

    def train_fake(self, surrogate_model, surrogate_trainer, fake_tensor, adv_opt,
                   temp_fake_user_tensor, verbose):
        opt = SGD(surrogate_model.parameters(), lr=self.look_ahead_lr)
        with higher.innerloop_ctx(surrogate_model, opt) as (fmodel, diffopt):
            fmodel.train()
            profiles = gumbel_topk(fake_tensor, self.n_inters, hard=False)
            normed_profiles = F.normalize(profiles, dim=1, p=1)
            normed_n_profiles = F.normalize(1 - profiles, dim=1, p=1)

            first_loss_fake = None
            for s in range(self.look_ahead_step):
                scores, l2_norm_sq = fmodel.forward(temp_fake_user_tensor, self.candidate_items)
                score_p = (scores * normed_profiles).sum(dim=1).detach()
                score_n = (scores * normed_n_profiles).sum(dim=1).detach()
                loss_p = F.softplus(score_n[:, None] - scores) + surrogate_trainer.l2_reg * l2_norm_sq
                loss_n = F.softplus(scores - score_p[:, None]) + surrogate_trainer.l2_reg * l2_norm_sq
                loss_p = (loss_p * profiles).sum()
                loss_n = (loss_n * normed_n_profiles * profiles.sum(dim=1, keepdim=True)).sum()
                diffopt.step(loss_p + loss_n)
                if first_loss_fake is None:
                    loss_p = loss_p / profiles.sum()
                    loss_n = loss_n / profiles.sum()
                    first_loss_fake = loss_p + loss_n
                vprint('Unroll step {:d}, Positive loss {:.6f}, Negative loss {:.6f}'.
                       format(s, loss_p.item(), loss_n.item()), verbose)

            fmodel.eval()
            target_scores, top_scores = self.get_target_item_and_top_scores(fmodel)
            adv_loss = goal_oriented_loss(target_scores, top_scores, self.expected_hr)
            consistency_loss = self.consistency_reg * first_loss_fake
            entropy_loss = self.entropy_reg * F.cross_entropy(normed_profiles, normed_profiles)
            l2_loss = self.l2_reg * torch.norm(fake_tensor, p=2)
            adv_grads = torch.autograd.grad(adv_loss + consistency_loss + entropy_loss + l2_loss, fake_tensor)[0]

        adv_opt.zero_grad()
        fake_tensor.grad = F.normalize(adv_grads, dim=1, p=2)
        adv_opt.step()
        vprint('Adversarial Loss: {:.6f}, Consistency Loss {:.6f}, Entropy Loss {:.6f}, L2 Loss {:.6f}'.
               format(adv_loss.item(), consistency_loss.item(), entropy_loss.item(), l2_loss.item()), verbose)

    def init_fake_tensor(self, n_temp_fakes):
        fake_tensor = torch.zeros([n_temp_fakes, self.candidate_items.shape[0]], dtype=torch.float32, device=self.device)
        fake_tensor[:, -self.target_items.shape[0]:] = 10.
        fake_tensor.requires_grad = True
        return fake_tensor

    def choose_filler_items(self, fake_tensor, temp_fake_user_tensor):
        with torch.no_grad():
            _, top_indices = fake_tensor.topk(self.n_inters, dim=1)
            n_inters = torch.gt(fake_tensor, 0.).int().sum(dim=1)
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            filler_indices = top_indices[u_idx, :n_inters[u_idx]]
            filler_items = self.candidate_items[filler_indices]

            f_u = temp_fake_user_tensor[u_idx]
            filler_items = filler_items.cpu().numpy().tolist()
            self.fake_user_inters[f_u - self.n_users] = filler_items
            self.dataset.train_data.append(set(filler_items))
            self.dataset.val_data.append({})
            self.dataset.train_array += [[f_u, item] for item in filler_items]
            self.dataset.n_users += 1
            self.fake_inters += len(filler_items)

    def retrain_surrogate(self, fake_tensor, adv_opt, temp_fake_user_tensor, fake_nums_str,
                          adv_epoch, verbose, writer):
        surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)
        for retraining_epoch in range(self.n_retraining_epochs):
            start_time = time.time()

            for _ in range(self.train_fake_its):
                self.train_fake(surrogate_model, surrogate_trainer, fake_tensor, adv_opt, temp_fake_user_tensor, verbose)
            surrogate_model.train()
            t_loss = self.train_rec(surrogate_trainer, fake_tensor)
            target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)

            consumed_time = time.time() - start_time
            vprint('Adversarial Epoch {:d}/{:d}, Retraining Epoch {:d}/{:d}, Time: {:.3f}s, '
                   'Train Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                   format(adv_epoch, self.n_adv_epochs, retraining_epoch, self.n_retraining_epochs, consumed_time,
                          t_loss, target_hr * 100.), verbose)
            global_retraining_epoch = adv_epoch * self.n_retraining_epochs + retraining_epoch
            writer_tag = '{:s}_{:s}'.format(self.name, fake_nums_str)
            if writer:
                writer.add_scalar(writer_tag + '/Train_Loss', t_loss, global_retraining_epoch)
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
            adv_opt = SGD([fake_tensor], lr=self.lr, momentum=self.momentum)

            for adv_epoch in range(self.n_adv_epochs):
                self.retrain_surrogate(fake_tensor, adv_opt, temp_fake_user_tensor, fake_nums_str,
                                       adv_epoch, verbose, writer)

            self.choose_filler_items(fake_tensor, temp_fake_user_tensor)
            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            print('Fake #{:s} has been generated! Time: {:.3f}s'.format(fake_nums_str, consumed_time))

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.fake_inters]
        self.dataset.n_users -= self.n_fakes
