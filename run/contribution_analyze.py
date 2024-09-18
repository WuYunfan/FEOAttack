from utils import init_run, get_target_items
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import torch
import os
import json
from dataset import get_dataset
from attacker import get_attacker
import shutil


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[0]
    attacker_config = get_attacker_config()[3]

    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset)
    print('Target items: {:s}'.format(str(target_items)))
    attacker_config['target_items'] = target_items
    attacker = get_attacker(attacker_config, dataset)

    if not os.path.exists('fake_user.json'):
        attacker.generate_fake_users()
        with open('data.json', 'w') as f:
            json.dump(attacker.fake_user_inters, f)
    else:
        with open('data.json', 'r') as f:
            attacker.fake_user_inters = json.load(f)

    model_copy = copy.deepcopy(model)
    optimizer_copy = type(optimizer)(model_copy.parameters(), lr=optimizer.defaults['lr'])
    optimizer_copy.load_state_dict(optimizer.state_dict())



if __name__ == '__main__':
    main()
