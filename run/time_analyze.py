from utils import init_run, get_target_items, AttackDataset, set_seed
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
plt.rc('font', family='Times New Roman')
plt.rcParams['pdf.fonttype'] = 42

def attack_eval(trainer, recalls):
    recalls.append(trainer.eval('attack')[1]['Recall'][trainer.topks[0]])

def analyze(recall_records):
    print(recall_records.shape)
    recall_diff = recall_records[:, 1:] - recall_records[:, :-1]
    diff_mean = np.mean(recall_diff, axis=0)
    diff_std = np.std(recall_diff, axis=0)
    epochs = np.arange(recall_diff.shape[1])

    pdf = PdfPages('time_analyze.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 3))
    ax1.text(20, 0.004, 'Hit ratio increases in early epochs', fontsize=18, color='#9a0700', fontweight='bold')
    ax1.plot(epochs[17:], diff_mean[17:], '-', marker='o', markersize=3, color='#8daadb', linewidth=2)
    ax1.fill_between(epochs[17:], diff_mean[17:] - diff_std[17:], diff_mean[17:] + diff_std[17:], alpha=0.2, color='#8daadb')
    ax1.plot(epochs[:18], diff_mean[:18], '-', marker='o', markersize=3, color='#9a0700', linewidth=2)
    ax1.fill_between(epochs[:18], diff_mean[:18] - diff_std[:18], diff_mean[:18] + diff_std[:18], alpha=0.2, color='#9a0700')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.6)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in', labelsize=14)
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.set_xlabel('Training Epoch', fontsize=16)
    ax1.set_ylabel('Hit Ratio Change', fontsize=16)
    ax1.set_ylim(-0.006, 0.008)
    ax1.set_yticks(np.arange(-0.006, 0.01, 0.002))
    ax1.set_xticks(np.arange(0, diff_mean.shape[0] + 2, 10))
    ax1.axhline(0, color='black', linewidth=0.8)

    correlations = []
    conf_intervals = []
    for i in range(recall_diff.shape[1]):
        p = pearsonr(recall_records[:, -1], recall_diff[:, i])
        correlations.append(p.statistic)
        conf_interval = [p.confidence_interval().low, p.confidence_interval().high]
        conf_intervals.append(conf_interval)
    correlations = np.array(correlations)
    conf_intervals = np.array(conf_intervals)
    ax2.text(20, 0.55, 'Hit ratio changes in early epochs\n correlate more with final results',
             fontsize=18, color='#9a0700', fontweight='bold')
    ax2.plot(epochs[18:], correlations[18:], '-', marker='o', markersize=3, color='#8daadb', linewidth=2)
    ax2.fill_between(epochs[18:], conf_intervals[18:, 0], conf_intervals[18:, 1], alpha=0.2, color='#8daadb')
    ax2.plot(epochs[:19], correlations[:19], '-', marker='o', markersize=3, color='#9a0700', linewidth=2)
    ax2.fill_between(epochs[:19], conf_intervals[:19, 0], conf_intervals[:19, 1], alpha=0.2, color='#9a0700')
    ax2.grid(True, which='major', linestyle='--', linewidth=0.6)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in', labelsize=14)
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.set_xlabel('Training Epoch', fontsize=16)
    ax2.set_ylabel('Pearson Coefficient', fontsize=16)
    ax2.set_ylim(-0.6, 1.)
    ax2.set_yticks(np.arange(-0.6, 1.2, 0.2))
    ax2.set_xticks(np.arange(0, correlations.shape[0] + 2, 10))
    ax2.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

def main():
    if os.path.exists('time_analyze.npy'):
        recall_records = np.load('time_analyze.npy')
        analyze(recall_records)
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
    recall_records = []
    for i, fake_inters in enumerate(fake_user_inters):
        set_seed(2023)
        attacker.fake_user_inters = [fake_inters for _ in range(attacker.n_fakes)]
        attacker.dataset = get_dataset(dataset_config)
        attacker.inject_fake_users()
        attacker.model = get_model(model_config, attacker.dataset)
        attacker.trainer = get_trainer(trainer_config, attacker.model)

        recalls = []
        extra_eval = (attack_eval, (recalls, ))

        attacker.trainer.train(extra_eval=extra_eval, verbose=False)
        recalls.append(attacker.trainer.eval('attack')[1]['Recall'][attacker.topk])
        recall_records.append(recalls)
        print('Finish evaluating fake user {:d}.'.format(i + 1))
    np.save('time_analyze.npy', np.array(recall_records))


if __name__ == '__main__':
    main()
