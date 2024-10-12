# Habitat-TorchRL-Prototype
Small prototype to show Habitat usage with TorchRL for visual deep reinforcement learning

# Setup
```shell
micromamba create -f environment.yaml -y
```
```shell
micromamba activate habitat
```
```shell
sudo apt install git-lfs
```
```shell
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data/
```
```shell
python explore_env.py
```