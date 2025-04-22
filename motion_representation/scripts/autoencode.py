import torch
import wandb.plot
from absl import app
from ml_collections import config_flags
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
import mujoco
from transforms3d.quaternions import mat2quat
# custom modules
from motion_representation.utils.data_loader import  get_K_data_loaders, get_data_loaders
from motion_representation.utils.utils import get_parameters, Config
from motion_representation.modules.transformer_vqvae import TransformerVQVAE
from motion_representation.scripts.config.ae_default_config import get_config
from motion_retarget.scripts.utils.viewer import MujocoViewer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
ae_config = get_config()
ae_cfg_flags = config_flags.DEFINE_config_dict('ae',ae_config)


class Trainer():
    def __init__(self, model, optimizer, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.optimizer = optimizer
        self.model = model
        self.current_epoch = 0
        self.current_iter = 0
        if self.cfg.visu_mujoco:
            xml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'motion_retarget/robots', f'{self.cfg.robot}', 'xml', 'plane.xml')
            self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            self.mujoco_model.opt.timestep = 0.005 # 200Hz
            self.dataset_hz = (1/50)
            


    def save_checkpoint(self,tag):
        # check if the directory exists
        save_dir = os.path.join(self.cfg.ckpt_dir,self.cfg.robot)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model, os.path.join(save_dir, f'model_{tag}.pt'))
    
    def visu(self,x_hat,x,tag=None, plt_tag=None, is_visu = True):
        assert len(x)==len(x_hat), print(" please check the input variable")
        time_line = np.arange(0,len(x),1)
        if is_visu:
            fig,axs = plt.subplots(3,6,figsize=(25,12))
            for i in range(3):
                for j in range(6):
                    axs[i,j].plot(time_line,x[:,j + i*6],label = 'gt')
                    axs[i,j].plot(time_line,x_hat[:,j + i*6],label='x_hat')
                    axs[i,j].legend()
                    axs[i,j].set_title(f'dim {j + i*6}')

            plt_tag = 'visu' if plt_tag is None else plt_tag
            if self.cfg.wandb:
                wandb.log({"visu":wandb.Image(plt)})
            plt.close(fig)


    @torch.no_grad()
    def evaluate_loss(self,train_data,val_data):
        self.model.eval()
        out = {}
        for split in ['train','val']:
            loader = train_data if split == 'train' else val_data
            losses = []
            for batch in loader:
                data = batch['data'].to(self.device) # B,T,D
                valid = batch['valid'].to(self.device)
                loss,x_hat = self.model(data,valid)
                labels = batch['motion_source']
                losses.append(loss.item())

            min_val = loader.dataset.min_val
            max_val = loader.dataset.max_val

            valid_mask = batch['valid'].to(self.device)
            out[split] = torch.tensor(losses).mean().item()
        self.model.train()
        return out, x_hat, data, valid_mask, min_val, max_val, labels# B,T,D, data

    def train_n_iters(self,data):
        for batch in data:
            data = batch['data'].to(self.device)
            valid = batch['valid'].to(self.device)
            loss,_ = self.model(data,valid)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                    losses, x_hat, x, valid_mask, min_val, max_val, labels = self.eval(train_data, val_data)
                    print(f"epoch: {epoch}, train loss: {losses['train']:.06f}, val loss: {losses['val']:.06f}")
        
                    print(f"lr: {self.optimizer.param_groups[0]['lr']}")
                    if self.cfg.wandb:
                        wandb.log({"train loss": losses['train'], "val loss": losses['val']})
                        
                    if self.cfg.visu_img:
                        till_valid = torch.sum(valid_mask, dim=1).squeeze(1).long()
                        min_val_ = train_data.dataset.min_val
                        max_val_ = train_data.dataset.max_val
                        x_hat = x_hat[-1, :till_valid[-1], :].squeeze(0).cpu().detach().numpy()
                        x = x[-1, :till_valid[-1], :].squeeze(0).cpu().detach().numpy()
                        # recover -1,1 minmax scale
                        x_hat = (x_hat + 1) * (max_val_ - min_val_) / 2 + min_val_
                        x = (x + 1) * (max_val_ - min_val_) / 2 + min_val_
                        self.visu(x_hat, x, tag=epoch)
        if self.cfg.wandb:
            assert wandb.run is not None
            self.save_checkpoint(tag= wandb.run.name)
            wandb.finish()
        else:
            self.save_checkpoint(tag='final')

    def eval(self,train,val):
        out = self.evaluate_loss(train,val)
        return out
    
    def visu_mujoco(self,x_hat):
            dt = 1/50
            mujoco_viewer = MujocoViewer(model=self.mujoco_model, dt=dt)
            self.mujoco_model.opt.timestep = 0.005
            m_fr = [10, 11, 12]
            m_fl = [7, 8,9]
            m_rr = [ 16, 17, 18]
            m_rl = [ 13, 14, 15]
            pin_fl = [9, 10, 11]
            pin_fr = [12, 13, 14]
            pin_rl = [15, 16, 17]
            pin_rr = [18, 19, 20]
            # for go2
            fl_mask = np.array([1, 1, 1])
            fr_mask = np.array([1, -1, -1])
            rl_mask = np.array([-1, 1, 1])
            rr_mask = np.array([-1, -1, -1])
            
            foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            foot_geom_indices = np.array([mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in foot_names])
            
            for repeat in range(1):
                qpos_buf = []
                self.mujoco_data.qpos = np.zeros(self.mujoco_model.nq)
                for i in range(len(x_hat)):
                    mat_xy = x_hat[i][3:9].reshape(3,2)
                    mat_z = np.cross(mat_xy[:,0],mat_xy[:,1])
              
                    mat = np.hstack([mat_xy,mat_z.reshape(-1,1)])
                    
                    quat = mat2quat(mat.flatten())
                    self.mujoco_data.qpos[3:7] = quat # root orientation
                    self.mujoco_data.qpos[m_fr] = x_hat[i][pin_fr] * fr_mask
                    self.mujoco_data.qpos[m_fl] = x_hat[i][pin_fl] * fl_mask
                    self.mujoco_data.qpos[m_rr] = x_hat[i][pin_rr] * rr_mask
                    self.mujoco_data.qpos[m_rl] = x_hat[i][pin_rl] * rl_mask
    
                    self.mujoco_data.qpos[:2] += x_hat[i][:2] * self.dataset_hz# root position
                    # remove height offset
                    self.mujoco_data.qpos[2] = x_hat[i][2]
                    q_pos = self.mujoco_data.qpos.copy()
                    for _ in range(4):
                        mujoco.mj_fwdPosition(self.mujoco_model, self.mujoco_data)
                    feet_print = self.mujoco_data.geom_xpos[foot_geom_indices]
                    mujoco_viewer.render(self.mujoco_data)
                    c_data = np.asarray(list(q_pos) + list(feet_print.flatten()))
                    c_data = c_data.tolist()
                    assert len(c_data) == 31
                    qpos_buf.append(c_data)
            # terminate mujoco viewer
            mujoco_viewer.stop()
            
            return np.asarray(qpos_buf)
    
    
    @torch.no_grad()
    def test(self,loader):
        self.model.eval()
        for iter,batch in enumerate(loader):
            x = batch['data'].to(self.device) # B,T,D
            valid_mask = batch['valid'].to(self.device)
            min_val = loader.dataset.min_val
            max_val = loader.dataset.max_val
            loss,x_hat = self.model(x,valid_mask)
            print(f'test loss :{loss}')
            till_valid = torch.sum(valid_mask, dim=1).squeeze(1).long()
            x_hat = x_hat[-1, :till_valid[-1], :].squeeze(0).cpu().detach().numpy()
            x = x[-1, :till_valid[-1], :].squeeze(0).cpu().detach().numpy()
            # recover -1,1 minmax scale
            x_hat = (x_hat + 1) * (max_val - min_val) / 2 + min_val
            x = (x + 1) * (max_val - min_val) / 2 + min_val
            if self.cfg.visu_img:
                self.visu(x_hat,x, tag = f'test_ae_{iter}',is_visu=False)
            if self.cfg.visu_mujoco:
                self.visu_mujoco(x_hat)
        self.model.train()



def main(_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ae_cfg = ae_cfg_flags.value
    torch.manual_seed(ae_cfg.seed)
    np.random.seed(ae_cfg.seed)
    k_train_loader, k_val_loader = get_K_data_loaders(batch_size=ae_cfg.train_batch_size)
    n_dims = k_train_loader[0].dataset[0]['data'].shape[-1]

    if ae_cfg.wandb:
        wandb.init(project=f"auto_encode_{ae_cfg.robot}")
        wandb.config.update(ae_cfg)
    
    if ae_cfg.load_pretrain or ae_cfg.state == 'val':
        if ae_cfg.pretrain_model_path is None:
            raise ValueError('please provide the pretrain model path')
        else:
        
            model_path = os.path.join(ae_cfg.ckpt_dir, ae_cfg.robot, ae_cfg.pretrain_model_path)
            model = torch.load(model_path, map_location=device)
            print(f'pretrain model {ae_cfg.pretrain_model_path} loaded !')
    else:
        TransformerVqvae_config = Config(in_dim = n_dims, out_dim=n_dims,
                                n_layers = ae_cfg.model_cfg.n_layers, n_embd = ae_cfg.model_cfg.n_embd, 
                                n_heads = ae_cfg.model_cfg.n_heads, block_size = ae_cfg.model_cfg.block_size,
                                attn_dropout = ae_cfg.model_cfg.attn_dropout, dropout = ae_cfg.model_cfg.dropout, 
                                pos_all = ae_cfg.model_cfg.pos_all, n_e = ae_cfg.model_cfg.n_e,  
                                e_dim= ae_cfg.model_cfg.e_dim, n_books = ae_cfg.model_cfg.n_books, balance = ae_cfg.model_cfg.balance)
      
        model = TransformerVQVAE(TransformerVqvae_config)

    model = model.to(device)
    print(f"Number of parameters: {get_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr = ae_cfg.learning_rate)
    train = Trainer(model,optimizer,device,ae_cfg)
    train.model = model
    if ae_cfg.state == 'train':
        train.fit(k_train_loader,k_val_loader)
    else:
        full_dataset = get_data_loaders('full',batch_size=1,shuffle=False)
        train.test(full_dataset)

if __name__ == '__main__':
    app.run(main)