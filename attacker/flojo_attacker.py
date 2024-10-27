import gc
import random
import torch
from torch.cuda import device
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, vprint, get_target_hr, goal_oriented_loss, PartialDataLoader
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
        self.step_user = attacker_config['step_user']
        self.item_interval = attacker_config['item_interval']
        self.n_retraining_epochs = attacker_config['n_retraining_epochs']
        self.lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']
        self.look_ahead_lr = attacker_config['look_ahead_lr']
        self.look_ahead_step = attacker_config['look_ahead_step']
        self.train_fake_its = attacker_config['train_fake_its']

        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=False)

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

    def train_fake(self, surrogate_model, surrogate_trainer, fake_tensor, adv_opt,
                   temp_fake_user_tensor, it, verbose):
        opt = SGD(surrogate_model.parameters(), lr=self.look_ahead_lr)
        with higher.innerloop_ctx(surrogate_model, opt) as (fmodel, diffopt):
            fmodel.train()
            profiles = F.gumbel_softmax(fake_tensor, hard=False, dim=1)
            filler_items = self.dataset.train_data[-temp_fake_user_tensor.shape[0]:]
            filler_items = [list(items) for items in filler_items]
            filler_items = torch.tensor(filler_items, dtype=torch.int64, device=self.device)
            profiles = torch.scatter(profiles, 1, filler_items, 1.)
            normed_p_profiles = F.normalize(profiles, dim=1, p=1)
            normed_n_profiles = F.normalize(1 - profiles, dim=1, p=1)

            for s in range(self.look_ahead_step):
                scores, l2_norm_sq = fmodel.forward(temp_fake_user_tensor)
                score_p = (scores * normed_p_profiles).sum(dim=1).detach()
                score_n = (scores * normed_n_profiles).sum(dim=1).detach()
                loss_p = F.softplus(score_n[:, None] - scores) + surrogate_trainer.l2_reg * l2_norm_sq
                loss_n = F.softplus(scores - score_p[:, None]) + surrogate_trainer.l2_reg * l2_norm_sq
                loss_p = (loss_p * profiles).sum()
                loss_n = (loss_n * normed_n_profiles * profiles.sum(dim=1, keepdim=True)).sum()
                diffopt.step(loss_p + loss_n)
                loss_p = loss_p / profiles.sum()
                loss_n = loss_n / profiles.sum()
                vprint('Unroll step {:d}, Positive loss {:.6f}, Negative loss {:.6f}'.
                       format(s, loss_p.item(), loss_n.item()), verbose)

            fmodel.eval()
            target_scores, top_scores = self.get_target_item_and_top_scores(fmodel)
            adv_loss = goal_oriented_loss(target_scores, top_scores, self.expected_hr)
            adv_grads = torch.autograd.grad(adv_loss, fake_tensor)[0]

        adv_opt.zero_grad()
        fake_tensor.grad = F.normalize(adv_grads, dim=1, p=2)
        adv_opt.step()
        vprint('Iteration {:d}: Adversarial Loss: {:.6f}'.format(it, adv_loss.item()), verbose)

    def init_fake_tensor(self, temp_fake_user_tensor):
        fake_tensor = torch.zeros([temp_fake_user_tensor.shape[0], self.n_items], dtype=torch.float32, device=self.device)
        fake_tensor[:, self.target_item_tensor] = 1.
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            f_u = temp_fake_user_tensor[u_idx]
            filler_items = self.fake_user_inters[f_u - self.n_users]
            fake_tensor[u_idx, torch.tensor(filler_items, dtype=torch.int64)] = -np.inf
        fake_tensor.requires_grad = True
        return fake_tensor

    def add_filler_items(self, fake_tensor, temp_fake_user_tensor):
        filler_items = fake_tensor.argmax(dim=1).cpu().numpy().tolist()
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            f_u = temp_fake_user_tensor[u_idx]
            self.fake_user_inters[f_u - self.n_users].append(filler_items[u_idx])
            self.dataset.train_data[f_u].add(filler_items[u_idx])
            self.dataset.train_array.append([f_u, filler_items[u_idx]])

    def retrain_surrogate(self, temp_fake_user_tensor, fake_nums_str, verbose, writer):
        surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)
        for retraining_epoch in range(self.n_retraining_epochs):
            start_time = time.time()

            fake_tensor = self.init_fake_tensor(temp_fake_user_tensor)
            adv_opt = SGD([fake_tensor], lr=self.lr, momentum=self.momentum)
            for it in range(self.train_fake_its):
                self.train_fake(surrogate_model, surrogate_trainer, fake_tensor, adv_opt, temp_fake_user_tensor, it, verbose)
            if retraining_epoch % self.item_interval == 0 and len(self.fake_user_inters[-1]) < self.n_inters:
                self.add_filler_items(fake_tensor, temp_fake_user_tensor)

            surrogate_model.train()
            t_loss = surrogate_trainer.train_one_epoch(None)
            target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)

            consumed_time = time.time() - start_time
            vprint('Retraining Epoch {:d}/{:d}, Time: {:.3f}s, Added items: {:d}, '
                   'Train Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                   format(retraining_epoch, self.n_retraining_epochs, consumed_time,
                          len(self.fake_user_inters[-1]), t_loss, target_hr * 100.), verbose)
            writer_tag = '{:s}_{:s}'.format(self.name, fake_nums_str)
            if writer:
                writer.add_scalar(writer_tag + '/Train_Loss', t_loss, self.n_retraining_epochs)
                writer.add_scalar(writer_tag + '/Hit_Ratio@' + str(self.topk), target_hr, self.n_retraining_epochs)

    def generate_fake_users(self, verbose=True, writer=None):
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step_user, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            start_time = time.time()
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating fake #{:s} !'.format(fake_nums_str))
            temp_fake_user_tensor = np.arange(fake_user_end_indices[i_step - 1],
                                              fake_user_end_indices[i_step]) + self.n_users
            temp_fake_user_tensor = torch.tensor(temp_fake_user_tensor, dtype=torch.int64, device=self.device)
            n_temp_fakes = temp_fake_user_tensor.shape[0]

            self.dataset.train_data += [set() for _ in range(n_temp_fakes)]
            self.dataset.val_data += [set() for _ in range(n_temp_fakes)]
            self.dataset.n_users += n_temp_fakes

            self.retrain_surrogate(temp_fake_user_tensor, fake_nums_str, verbose, writer)

            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            print('Fake #{:s} has been generated! Time: {:.3f}s'.format(fake_nums_str, consumed_time))

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
