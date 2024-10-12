# Habitat-TorchRL-Prototype
Small prototype to show Habitat usage with TorchRL for visual deep reinforcement learning

# Setup
```shell
sudo apt install git-lfs
```
```shell
micromamba create -f environment.yaml -y
```
```shell
micromamba activate habitat
```
```shell
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data/
```
```shell
python explore_env.py
```

## Other tasks   
```shell
python -m habitat_sim.utils.datasets_download --uids replica_cad --data-path data/
```
