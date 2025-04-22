from ml_collections import config_dict
from pathlib import Path

def get_config():
    config = config_dict.ConfigDict()

    # General training parameters
    config.robot = 'go2' # 'a1' or 'go2'
    config.max_epochs = 400
    config.learning_rate = 1e-3
    config.train_batch_size = 32
    config.val_batch_size = 4
    config.eval_freq = 10


    # Training helper parameters
    config.wandb = 0  # 1 to use WandB, 0 otherwise
    config.visu_img = 0   # 1 to enable visualization, 0 otherwise
    config.state = 'train'  # Can be 'train' or 'val'
    config.ckpt_dir = str(Path(__file__).resolve().parents[1].joinpath('logs/gpt_model/'))
    config.load_pretrain = 0  # 1 to load pre-trained model, 0 otherwise
    config.pretrain_model_path = str(Path(__file__).resolve().parents[1].joinpath('logs/model/model_mild-sweep-255.pt'))
    config.save_traj = 0  # 1 to save trajectories, 0 otherwise

    # Visualization parameter
    config.visu_mujoco = 0  # 1 to enable Mujoco visualization, 0 otherwise

    # notes
    config.notes = 'test'


    # AE (Autoencoder) Configuration
    config.ae_model = config_dict.ConfigDict()
    config.ae_model.name = "model_final" # Name of the AE model

    # GPT (Generative Pre-trained Transformer) Configuration
    config.model_cfg = config_dict.ConfigDict()
    config.model_cfg.tok_dim = 128
    config.model_cfg.dropout = 0.1
    config.model_cfg.n_layers = 1
    config.model_cfg.block_size = 1024
    config.model_cfg.n_heads = 4
    config.model_cfg.autoreg_head = 1
    config.model_cfg.enable_motion_source = 1
    config.model_cfg.max_sample_len = 2
    config.model_cfg.sample_len = 2 # in seconds


    return config
