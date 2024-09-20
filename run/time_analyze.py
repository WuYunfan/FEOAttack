from utils import init_run, get_target_items, AttackDataset
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import torch
import os
import json
from dataset import get_dataset
from attacker import get_attacker
import shutil
import copy
from model import get_model
from trainer import get_trainer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

def contribution_eval(trainer, profiles, n_users, contributions):
    trainer = copy.deepcopy(trainer)
    before_recall = trainer.eval('attack')[1]['Recall'][trainer.topks[0]]

    attack_dataset = AttackDataset(profiles, n_users, trainer.negative_sample_ratio)
    attack_dataloader = DataLoader(attack_dataset, batch_size=trainer.dataloader.batch_size,
                                   num_workers=trainer.dataloader.num_workers,
                                   persistent_workers=False, pin_memory=False)
    trainer.dataloader = attack_dataloader
    trainer.train_one_epoch(None)
    after_recall = trainer.eval('attack')[1]['Recall'][trainer.topks[0]]
    contributions.append(after_recall - before_recall)

def analyze(final_recalls, contribution_records):
    print(final_recalls.shape)
    print(contribution_records.shape)

    pdf = PdfPages('contribution_epoch.pdf')
    fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 4))
    ax.plot(contribution_records.mean(axis=0))
    ax.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    correlations = []
    for i in range(contribution_records.shape[1]):
        correlations.append(pearsonr(final_recalls, contribution_records[:, i]).statistic)
    print(correlations[:10])
    pdf = PdfPages('contribution_sim.pdf')
    fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 4))
    ax.plot(correlations)
    ax.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    pdf.savefig()
    plt.close(fig)
    pdf.close()

def main():
    if os.path.exists('time_analyze.npz'):
        data = np.load('time_analyze.npz')
        final_recalls, contribution_records = data['arr_0'], data['arr_1']
        analyze(final_recalls, contribution_records)
        return

    log_path = __file__[:-3]
    init_run(log_path, 2023)

    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[0]
    trainer_config['n_epochs'] = 100
    attacker_config = get_attacker_config()[4]

    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset)
    print('Target items: {:s}'.format(str(target_items)))
    attacker_config['target_items'] = target_items
    attacker = get_attacker(attacker_config, dataset)

    if not os.path.exists('fake_user_inters.json'):
        attacker.generate_fake_users(verbose=False)
        with open('fake_user_inters.json', 'w') as f:
            json.dump(attacker.fake_user_inters, f)
    else:
        with open('fake_user_inters.json', 'r') as f:
            attacker.fake_user_inters = json.load(f)
    print('Fake users have been generated!')

    fake_user_inters = attacker.fake_user_inters
    final_recalls = []
    contribution_records = []
    for i, fake_inters in enumerate(fake_user_inters):
        attacker.fake_user_inters = [fake_inters for _ in range(attacker.n_fakes)]
        attacker.dataset = get_dataset(dataset_config)
        attacker.inject_fake_users()
        attacker.model = get_model(model_config, attacker.dataset)
        attacker.trainer = get_trainer(trainer_config, attacker.model)

        profiles = torch.zeros([attacker.n_fakes, attacker.n_items], device=device, dtype=torch.float)
        for f_idx in range(attacker.n_fakes):
            train_items = list(attacker.dataset.train_data[attacker.n_users + f_idx])
            train_items = torch.tensor(train_items, device=device, dtype=torch.int64)
            profiles[f_idx, train_items] = 1.
        contributions = []
        extra_eval = (contribution_eval, (profiles, attacker.n_users, contributions))

        attacker.trainer.train(extra_eval=extra_eval, verbose=False)
        final_recalls.append(attacker.trainer.eval('attack')[1]['Recall'][attacker.topk])
        contribution_records.append(contributions)
        print('Finish evaluating fake user {:d}.'.format(i + 1))
    np.savez('time_analyze.npz', np.array(final_recalls), np.array(contribution_records))


if __name__ == '__main__':
    main()
