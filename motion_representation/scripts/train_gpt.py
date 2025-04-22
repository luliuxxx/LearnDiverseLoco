import torch
import os
import wandb
from ml_collections import config_flags
from absl import app
import numpy as np
# custom modules
from motion_representation.utils.data_loader import  get_K_data_loaders
from motion_representation.utils.utils import get_parameters, Config
from motion_representation.modules.gpt import GPT
from motion_representation.scripts.config.gpt_default_config import get_config
from motion_representation.scripts.autoencode import Trainer

gpt_config = get_config()
gpt_cfg_flags = config_flags.DEFINE_config_dict('gpt', gpt_config)


class GTrainer(Trainer):
    def __init__(self, model, autoencoder,optimizer, device, cfg):
        super().__init__(model, optimizer, device, cfg)
        self.cfg = cfg
        self.device = device
        self.optimizer = optimizer
        self.model = model
        self.current_epoch = 0
        self.current_iter = 0
        self.autoencoder = autoencoder

    @torch.no_grad()
    def evaluate_loss(self,train_data,val_data):
        self.model.eval()
        out = {}
        for split in ['train','val']:
            loader = train_data if split == 'train' else val_data
            losses = []
            for batch in loader:
                data = batch['data'].to(self.device) # B,T,D
                valid_mask = batch['valid'].to(self.device)
                _,idx,_ = self.autoencoder.forward_latents(data,valid_mask)
                action = batch['motion_type'].to(self.device)
                source = batch['motion_source'].to(self.device)
                duration = batch['duration'].to(self.device)

                loss,_ = self.model(idx,action,source, duration)

                losses.append(loss.item())
            out[split] = torch.tensor(losses).mean().item()
        self.model.train()
        return out# B,T,D, data

    def train_n_iters(self,data):
        for batch in data:
            data = batch['data'].to(self.device)
            valid = batch['valid'].to(self.device)
            
            action = batch['motion_type'].to(self.device)
            source = batch['motion_source'].to(self.device)
            duration = batch['duration'].to(self.device)
            
            _,idx,_ = self.autoencoder.forward_latents(data,valid)

            loss,_ = self.model(idx,action,source, duration)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.current_iter += 1

    def fit(self, k_train_data, k_val_data):
        self.current_epoch = 0
        self.current_iter = 0
        self.model.train()
        while self.current_epoch <= self.cfg.max_epochs:
            epoch = self.current_epoch
            print(f"\n EPOCH={epoch:03d}/{self.cfg.max_epochs} - ITER={self.current_iter}")
            for train_data, val_data in zip(k_train_data, k_val_data):
                self.train_n_iters(train_data)
                self.current_epoch += 1
                if epoch % self.cfg.eval_freq == 0:
                    losses= self.eval(train_data, val_data)
                    print(f"epoch: {epoch}, train loss: {losses['train']:.06f}, val loss: {losses['val']:.06f}")
                    print(f"lr: {self.optimizer.param_groups[0]['lr']}")
                    if self.cfg.wandb:
                        wandb.log({"train loss": losses['train'], "val loss": losses['val']})
        if self.cfg.wandb:
            assert wandb.run is not None
            self.save_checkpoint(tag= wandb.run.name)
            wandb.finish()
        else:
            self.save_checkpoint(tag= 'final_')

    def eval(self,train,val):
        out = self.evaluate_loss(train,val)
        return out
    
    @torch.no_grad()
    def test(self,action,source,test_loader,src, txt_action, nr_traj = 0):
        self.model.eval()
        idx_hat = self.model.sample(action,source)
        x_hat_,valid_hat_ = self.autoencoder.forward_from_indices(idx_hat)
        x_hat_buf = x_hat_
        valid_hat_buf = valid_hat_

        x_hat = x_hat_buf
        valid_hat = valid_hat_buf
        min_val = test_loader.dataset.min_val
        max_val = test_loader.dataset.max_val
        x = x_hat.clone()
        till_valid = torch.sum(valid_hat, dim=1).long()
        x = x[-1,:,:].squeeze(0).cpu().detach().numpy()
        x_hat = x_hat[-1,:,:].squeeze(0).cpu().detach().numpy()
        x_hat = (x_hat + 1) * (max_val - min_val) / 2 + min_val
        x = (x + 1) * (max_val - min_val) / 2 + min_val
        if self.cfg.visu_img:
            self.visu(x_hat,x,tag=f'{nr_traj}_gpt',is_visu=False)
        if self.cfg.visu_mujoco:
            qbuf = self.visu_mujoco(x_hat)
            if self.cfg.save_traj:
                source2label = test_loader.dataset.source2label
                gait2label = test_loader.dataset.gait2label
                self.save_traj(qbuf,src,txt_action, nr_traj,source2label,gait2label)
               

    
    def save_traj(self,qpos_buf,src,action, nr_traj, source2label, gait2label):
        # save_path = os.getcwd() + f'/logs/gpt_traj/{self.cfg.robot}/{src}_{action}'
        save_path = os.getcwd() + f'/logs/gpt_traj/{self.cfg.robot}'
        order = ["x, y, z","wxyz","FL,FR,RL,RR, hip,thigh,calf","FL,FR,RL,RR,toes"] # 3 + 4 + 12 = 19 + 12 = 31
        # 
        assert qpos_buf.shape[1] == 31
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(save_path + f'/{src}_{action}_{nr_traj}.npz', data = qpos_buf, order = order, hz = 50, source2label = source2label, gait2label = gait2label)


def exponential_filter(data, alpha = 0.99):
    # data: T,D
    data_buf = np.zeros_like(data)
    data_buf[0] = data[0]
    for i in range(1,len(data)):
        data_buf[i] = alpha * data_buf[i-1] + (1-alpha) * data[i]
    return data_buf




def main(_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gpt_cfg = gpt_cfg_flags.value

    k_train_loader, k_val_loader = get_K_data_loaders(batch_size=gpt_cfg.train_batch_size)

    n_actions = k_train_loader[0].dataset.n_actions
    n_sources = k_train_loader[0].dataset.n_sources

    # sweep the gpt model, the gpt cfg should from wandb config

    vqvae_model_path = os.getcwd() + f'/scripts/logs/ae_model/{gpt_cfg.robot}/{gpt_cfg.ae_model.name}.pt'
    ae_model = torch.load(vqvae_model_path, map_location=device, weights_only=False)
    ae_model = ae_model.to(device)
    ae_model.eval()
    print(f"{vqvae_model_path} VQVAE model loaded !")
    print(f" Number of Codebooks: {ae_model.n_books}, coebook size: {ae_model.n_e//ae_model.n_books}")
    print(f"Number of parameters in AE model: {get_parameters(ae_model):,}")
    # freeze the vqvae model
    for param in ae_model.parameters():
        param.requires_grad = False

    if gpt_cfg.state == 'train':
        if gpt_cfg.wandb:
            wandb.init(project=f'gpt_train_{gpt_cfg.robot}', config=gpt_cfg)

        gpt_config = Config(n_e = ae_model.n_e, n_books = ae_model.n_books, 
                            n_actions = n_actions, n_sources = n_sources, tok_dim = gpt_cfg.model_cfg.tok_dim, dropout = gpt_cfg.model_cfg.dropout, 
                            n_layers = gpt_cfg.model_cfg.n_layers, block_size = gpt_cfg.model_cfg.block_size, n_heads = gpt_cfg.model_cfg.n_heads, autoreg_head = gpt_cfg.model_cfg.autoreg_head,
                            enable_motion_source = gpt_cfg.model_cfg.enable_motion_source, max_sample_len = gpt_cfg.model_cfg.max_sample_len, pred_len = gpt_cfg.model_cfg.sample_len)
        
        gpt_model = GPT(gpt_config)

        gpt_model = gpt_model.to(device)
        print(f"Number of parameters in gpt model: {get_parameters(gpt_model):,}")
    else:
        if gpt_cfg.pretrain_model_path:
            gpt_model_path = os.getcwd() + f'/scripts/logs/gpt_model/{gpt_cfg.robot}/{gpt_cfg.pretrain_model_path}'
            gpt_model = torch.load(gpt_model_path, map_location=device, weights_only=False)
            gpt_model.pred_len = gpt_cfg.model_cfg.sample_len
            print(f"Number of parameters in gpt model: {get_parameters(gpt_model):,}")
            print(f'pretrain model {gpt_cfg.pretrain_model_path} loaded !')
        else:
            raise ValueError('please provide the pretrain model path')
        
    # define the optimizer
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr = gpt_cfg.learning_rate)

    train = GTrainer(gpt_model,ae_model,optimizer,device,gpt_cfg)
    train.model = gpt_model
   
    if gpt_cfg.state == 'train':
            train.fit(k_train_loader,k_val_loader)
    else:
        # currently only the following source and gaits are supported
        # source: horse, dog, mpc, solo8
        # gait: walk, trot, pace, gallop, bound, crawl, stilt, leap, wave
        source = 'solo8'
        gait = 'trot'
        
        test_source = torch.tensor([k_train_loader[0].dataset.source2label[source]]).to(device)
        test_action = torch.tensor([k_train_loader[0].dataset.gait2label[gait]]).to(device)
        num_traj = 24
        print("testing the gpt model")
        for i in range(num_traj):
            print(f"Trajectory {i}")
            train.test(test_action,test_source,k_train_loader[0], source,  gait, nr_traj = i)


    
if __name__ == '__main__':
    app.run(main)