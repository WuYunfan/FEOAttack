import os.path
import torch
from IPython.testing.tools import fake_input

from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items, set_seed
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import shutil
import numpy as np

""""
target_items_lists = [[13163, 9306, 11375, 4780, 9990], [275, 7673, 7741, 10376, 7942],
                      [5851, 11920, 12563, 1254, 9246], [1692, 8460, 8293, 2438, 4490],
                      [12094, 12757, 3592, 4019, 2534]]

target_items_lists = [[13971, 24204, 10290, 24038, 9836], [19230, 5616, 557, 19986, 17702],
                      [21356, 5766, 2076, 4267, 18261], [8893, 20936, 19034, 16248, 178],
                      [21077, 10796, 4749, 19918, 5106]]                                  
"""


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    seed_list = [2024, 42, 0, 131, 1024]

    device = torch.device('cuda')
    dataset_config = get_config(device)[0][0]
    attacker_config = get_attacker_config()[-2]
    attacker_config['validate_topk'] = 100

    for i in range(5):
        set_seed(seed_list[i])
        dataset = get_dataset(dataset_config)
        target_items = get_target_items(dataset)
        print('Target items of {:d}th run: {:s}'.format(i, str(target_items)))
        attacker_config['target_items'] = target_items

        attacker = get_attacker(attacker_config, dataset)
        attacker.generate_fake_users()
        _, model_config, trainer_config = get_config(device)[0]
        attacker.eval(model_config, trainer_config)

        sims = []
        fake_users = np.arange(attacker.n_users, attacker.n_users + attacker.n_fakes)
        with torch.no_grad():
            scores = attacker.model.predict(fake_users)
            for u_idx in range(attacker.n_fakes):
                a = set(scores[u_idx].topk(100).indices.cpu().numpy().tolist())
                b = attacker.recommendation_list[u_idx]
                jaccard_sim =  len(a & b) / len(a | b)
                sims.append(jaccard_sim)
        print('Jaccard similarity between top-100 predicted items: {:.6f}'.format(np.mean(sims)))
        shutil.rmtree('checkpoints')


if __name__ == '__main__':
    main()
