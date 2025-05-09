import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items, set_seed
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import shutil
import numpy as np
target_items_lists = [[13163, 9306, 11375, 4780, 9990], [275, 7673, 7741, 10376, 7942],
                      [5851, 11920, 12563, 1254, 9246], [1692, 8460, 8293, 2438, 4490],
                      [12094, 12757, 3592, 4019, 2534]]
'''
target_items_lists = [[13971, 24204, 10290, 24038, 9836], [19230, 5616, 557, 19986, 17702],
                      [21356, 5766, 2076, 4267, 18261], [8893, 20936, 19034, 16248, 178],
                      [21077, 10796, 4749, 19918, 5106]]  

target_items_lists = [[39810, 127955, 192022, 34919, 192829], [109596, 156311, 138822, 143188, 29913]]
'''
target_items_lists = np.array(target_items_lists)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    seed_list = [2024, 42, 0, 131, 1024]

    device = torch.device('cuda')
    dataset_config = get_config(device)[0][0]

    hyperparameters = {'adv_weight': [1., 0.1, 0.01, 0.001, 0.0001, 0.],
                       'kl_weight': [1., 0.1, 0.01, 0.001, 0.0001, 0.],
                       'filler_limit': [1, 2, 5, 10], # [1, 2, 5, 10, 100]
                       'step_user': [5, 10, 20, 50, 100], # [50, 100, 200, 500, 1000]
                       'expected_hr': [0.01, 0.05, 0.1, 0.2, 0.5, 1.]}
    for key in hyperparameters.keys():
        for value in hyperparameters[key]:
            attacker_config = get_attacker_config()[-2]
            attacker_config[key] = value

            recalls = []
            for i in range(len(target_items_lists)):
                set_seed(seed_list[i])
                dataset = get_dataset(dataset_config)
                target_items = target_items_lists[i]
                print('Target items of {:d}th run: {:s}'.format(i, str(target_items)))
                attacker_config['target_items'] = target_items

                attacker = get_attacker(attacker_config, dataset)
                attacker.generate_fake_users()
                _, model_config, trainer_config = get_config(device)[0]
                recall = attacker.eval(model_config, trainer_config, detect=False) * 100
                recalls.append(recall)
                shutil.rmtree('checkpoints')
            print('Hyper-parameter {:s} with value {:.6f}, mean {:.3f}%, std {:.3f}%'.
                  format(key, value, np.mean(recalls), np.std(recalls, ddof=1)))


if __name__ == '__main__':
    main()