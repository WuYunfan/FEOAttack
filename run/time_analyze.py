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

def attack_eval(trainer, recalls):
    recalls.append(trainer.eval('attack')[1]['Recall'][trainer.topks[0]])

def analyze(recall_records):
    print(recall_records.shape)

    pdf = PdfPages('recall_epoch.pdf')
    fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 4))
    ax.plot(recall_records.mean(axis=0))
    ax.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    pdf.savefig()
    plt.close(fig)
    pdf.close()

def main():
    if os.path.exists('time_analyze.npy'):
        recall_records = np.load('time_analyze.npz')
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
