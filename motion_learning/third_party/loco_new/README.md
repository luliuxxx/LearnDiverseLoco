## Installation
1. Install RL-X

Default installation for a Linux system with a NVIDIA GPU.
For other configurations, see the RL-X [documentation](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide).
```bash
conda create -n loco_new python=3.11.4
conda activate loco_new
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch>=2.2.1" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

2. Install the project
```bash
git clone git@github.com:nico-bohlinger/loco_new.git
cd loco_new
pip install -e .
```


## Experiments on cluster
1. Setup conda environment and installation as described in the installation section
2. Run the following commands to start an experiment
```bash
cd loco_new/experiments
sbatch slurm_experiment.sh
```


## Local development
1. Create train.sh file in the experiments folder (all .sh files besides slurm_experiment.sh are ignored by git)
```bash
cd loco_new/experiments
touch train.sh
```
2. Add the following content to the train.sh file
```bash
python experiment.py \
    --algorithm.name=ppo.default \
    --environment.name="unitree_a1" \
    --runner.track_console=True \
    --algorithm.evaluation_frequency=-1 \
    --environment.render=False
```
3. Run the following command to start the experiment
```bash
cd loco_new/experiments
bash train.sh
```


## Testing a trained model
1. Create test.sh file in the experiments folder (all .sh files besides slurm_experiment.sh are ignored by git)
```bash
cd loco_new/experiments
touch test.sh
```
2. Add the following content to the test.sh file
```bash
python experiment.py \
    --algorithm.name=ppo.default \
    --environment.name="unitree_a1" \
    --environment.mode=test \
    --environment.render=True \
    --environment.add_goal_arrow=True \
    --runner.mode=test \
    --runner.load_model=model_best_jax
```
#### Controlling the robot
Either create commands.txt file
```bash
cd loco_new/experiments
touch commands.txt
```
And add the following content to the commands.txt file. Where the values are target x, y and yaw velocities
```bash
1.0
0.0
0.0
```

Or connect a **Xbox 360** controller and control the target x,y velocity with the left joystick and the yaw velocity with the right joystick.
