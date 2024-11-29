## Environment

Ensure the following dependencies are installed:

- **Python** >= 3.11
- **PyTorch** >= 2.1.0
- **DGL** >= 1.1.2

## Dataset

The processed datasets for **Gowalla**, **Yelp**, and **Amazon** can be downloaded from the following sources:

- [Baidu Wangpan](https://pan.baidu.com/s/1R18hvHsgwrrioKpfdf7UEA?pwd=akxv)
- [Google Drive](https://drive.google.com/file/d/1CWAxes6xNE3OJIph1YyeLhZmptYl5_XZ/view?usp=sharing)

Once downloaded, organize the data as follows:

```
├── FEOAttack-master
│   ├── data
│   │   ├── Gowalla
│   │   │   ├── time
│   │   │   ├── partial
│   │   ├── Yelp
│   │   ├── Amazon
│   ├── run
│   ├── attacker
│   ├── model.py
│   └── ...
```

### Notes on Dataset Structure

- **Gowalla Dataset:**
  - The `time` folder contains datasets split by time, including train and validation sets to facilitate training and hyperparameter tuning for victim recommenders.
  - The `partial` folder contains a subset of the `time` folder, designed for running attackers with partial knowledge.

- **Yelp and Amazon Datasets:**
  - These datasets are similarly structured but do not include the `partial` folder.

---

## Quick Start

### Running Experiments

To launch the experiment, use the following command:

```sh
python -u -m run.run
```

All baseline methods are implemented within **a unified framework** for ease of use.

### Testing on Different Datasets

To test on a different dataset, modify the import statements in `run/run.py`:

```python
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
```

Replace `'gowalla'` with the desired dataset (e.g., `yelp` or `amazon`).

### Running Baseline Attackers

To run a baseline attacker, update the following line in `run/run.py`:

```python
attacker_config = get_attacker_config()[0]
```

Change the index `0` to the index of the specific attacker configuration defined in `config.py`.

### Attacking Different Victim Recommenders

To attack a different victim recommender, modify the following line in `run/run.py`:

```python
configs = get_config(device)
```

Add the index of the desired victim recommender:

```python
configs = get_config(device)[x]
```

Here, `x` corresponds to the index of the specific victim recommender defined in `config.py`.

---

## Hyperparameter Tuning

The hyperparameters for all methods can be easily adjusted in the `config.py` file. Update the relevant sections to fine-tune your experiments.

---

## Folder Structure Overview

Here’s a quick overview of the key folders and files:

- **data/**: Contains the dataset folders (`Gowalla`, `Yelp`, `Amazon`).

- **run/**: Contains the main scripts to execute experiments.

- **attacker/**: Holds the implementations of attack methods.

- **model.py**: Includes the definitions of recommendation models.

- **config.py**: Specifies settings for datasets, recommenders and attackers.

- **trainer.py**: Implements the logic for training and evaluating recommendation models.

- **dataset.py**: Handles reading, preprocessing, and managing datasets.