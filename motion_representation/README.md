## Motion Representation
The motion representation module involves motion reconstruction and motion generation. This framework is adapted from the following sources:
- **Code:** [PoseGPT GitHub](https://github.com/naver/PoseGPT)  
- **Paper:** [PoseGPT Paper](https://arxiv.org/pdf/2210.10542)

---

## Robots

- **Unitree Go2**

---

Source the modules
```bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/modules"
```

##  Running the Autoencoder (VQ-VAE)
### Train
To train the autoencoder, use the following command:
```bash
python scripts/autoencode.py  --ae.state='train' --ae.visu_mujoco=0 --ae.wandb=1
```
### Test
To test the autoencoder, use the following command:
```bash
python scripts/autoencode.py --ae.state='val' --ae.visu_mujoco=1 --ae.wandb=0 --ae.pretrain_model_path='model_final.pt'
```

---

##  Running the Generator (GPT-based)
The following commands are used for working with the GPT-based generator:
###  Train
To train the GPT model, use:
```bash
python train_gpt.py --gpt.visu_img=1 --gpt.state='train' --gpt.visu_mujoco=0 --gpt.wandb=0
```

###  Test
To test the GPT model, use the following command:
```bash
python scripts/train_gpt.py --gpt.state='val' --gpt.visu_mujoco=1 --gpt.pretrain_model_path='model_final.pt' --gpt.save_traj=1
```
