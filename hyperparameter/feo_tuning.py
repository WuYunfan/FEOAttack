from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config as get_config
import torch
from dataset import get_dataset
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import shutil
import numpy as np


def objective(trial):
    adv_weight = trial.suggest_categorical('adv_weight', [0.003, 0.001, 0.0003])
    # [0.03, 0.01, 0.003] for gowalla, [0.1, 0.03, 0.01] for yelp
    # [0.003, 0.001, 0.0003] for gowalla-lgcn, [0.01, 0.003, 0.001] for yelp-lgcn, [0.03, 0.01, 0.003] for amazon-lgcn
    diverse_weight = trial.suggest_categorical('diverse_weight', [0.001, 0.0003, 0.0001])
    # [0.003, 0.001, 0.0003] for gowalla, [0.1, 0.03, 0.01] for yelp
    # [0.003, 0.001, 0.0003] for gowalla-lgcn, [3.e-5, 1.e-5, 0.] for yelp-lgcn, [0.01, 0.0003, 0.0001] for amazon-lgcn
    l2_weight = trial.suggest_categorical('l2_weight', [0.003, 0.001, 0.0003])
    # [0.01, 0.003, 0.001] for gowalla, [0.3, 0.1, 0.03] for yelp
    # [0.001, 0.0003, 0.0001] for gowalla-lgcn, [3.e-5, 1.e-5, 0.] for yelp-lgcn, [3.e-5, 1.e-5, 0.] for amazon-lgcn
    set_seed(2023)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[0]
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.001,
                                'n_epochs': 0, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False}
    attacker_config = {'name': 'FEOAttacker', 'n_fakes': 131, 'topk': 50, 'n_inters': 41,
                       'step_user': 10, 'n_training_epochs': 10, 'expected_hr': 0.05,
                       'adv_weight': adv_weight, 'diverse_weight': diverse_weight, 'l2_weight': l2_weight,
                       'look_ahead_lr': 0.1, 'prob': 0.9,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}

    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset, bottom_ratio=0.01)
    attacker_config['target_items'] = target_items
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users(verbose=False)
    recall = attacker.eval(model_config, trainer_config)
    shutil.rmtree('checkpoints')
    return recall


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'feo-tuning'
    storage = optuna.storages.RDBStorage(url='sqlite:///../{}.db'.format(study_name))
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.BruteForceSampler())

    study.optimize(objective)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    main()