import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import queue
import threading
"adpated from: https://github.com/CompVis/taming-transformers/taming/modules/vqvae/quantize.py"

def L2_efficient(x,y):
    return (x.pow(2).sum(1,keepdim=True) - 2*x@y + y.pow(2).sum(0,keepdim=True))

class EmaCodebookMeter:
    def __init__(self, codebook_size, ema_alpha=0.05):
        self.codebook_size = codebook_size
        self.bins = (torch.ones((self.codebook_size), requires_grad=False) / self.codebook_size).detach().cuda()
        self.ema_alpha = ema_alpha
        self.iters = 0
    
    def bincount(self, val, weights=None):
        norm = val.shape[0]
        weights = weights.reshape(-1) if weights is not None else None
        count = torch.bincount(val.reshape(-1), minlength=self.codebook_size, weights=weights).detach()
        self.iters += 1
        return count/norm
    
    def load(self,bins):
        self.bins = torch.tensor(bins, requires_grad=False).detach().cuda()
    
    def update(self, val, weights=None, n=1):
        count = self.bincount(val,weights=weights)
        alpha = max(self.ema_alpha, 1/(self.iters+1))
        self.bins = (1. - alpha) * self.bins + alpha * count
    
    def get_hist(self):
        return self.bins


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_e = config.n_e if hasattr(config, 'n_e') else 512 
        self.e_dim = config.e_dim if hasattr(config, 'e_dim') else 256
        self.beta = config.beta if hasattr(config, 'beta') else 0.25
        self.nbooks = config.n_books if hasattr(config, 'n_books') else 1
        self.balance = config.balance if hasattr(config, 'balance') else False
        # self.training = Fasle


        assert self.n_e % self.nbooks == 0, 'n_e must be divisible by nbooks'
        # make sure each book has the same number of embeddings
        self.n_e_i = self.n_e // self.nbooks
        # divide the embedding dimensions equally among the books
        
        embed_dims = (self.nbooks - 1)*[self.e_dim // self.nbooks] + [self.e_dim - (self.nbooks - 1)*(self.e_dim // self.nbooks)]
        self.embed_dims = embed_dims

        self.embeddings = torch.nn.ModuleDict({str(i):torch.nn.Embedding(self.n_e_i, d) for i,d in enumerate(self.embed_dims)})

        
        self.trackers = {}
        for i,e in self.embeddings.items():
            # initialize the embeddings
            e.weight.data.uniform_(-1/self.n_e_i, 1/self.n_e_i)
            self.trackers[int(i)] = EmaCodebookMeter(self.n_e_i)
            print(f"Codebook {i}: {list(e.weight.size())}")
        self.codebook_init_weights = [e.weight.clone().detach().cpu().numpy() for e in self.embeddings.values()]


    def get_state(self):
        return {i: self.trackers[i].get_hist().cpu().data.numpy() for i in self.trackers.keys()}
    
    def load_state(self,bins):
        for i,b in bins.items():
            self.trackers[i].load(b)
    
    def get_hist(self, i):
        return self.trackers[i].get_hist()
    
    def reset(self,i):
        for i in self.trackers.keys():
            self.trackers[i] = EmaCodebookMeter(self.n_e_i)

    def track_assigment(self,entries,i):
        self.trackers[i].update(entries)
    
    def forward_one(self, z, i, weights = None):
        bsize = self.e_dim // self.nbooks
        # z = z.reshape(-1,bsize)
        e_dim = bsize if i < self.nbooks - 1 else self.e_dim - (self.nbooks - 1)*bsize

        z_flattened = z.view(-1,e_dim) # B*T, divided_dims
        dist = L2_efficient(z_flattened, self.embeddings[str(i)].weight.t())
    
        if self.balance and weights is not None:
            wdist = dist * weights.unsqueeze(0)
            dist = -torch.nn.functional.softmax(-wdist, 1)
        
        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e_i).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        

        if self.balance and weights is not None:
            self.track_assigment(min_encoding_indices.detach(), i)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embeddings[str(i)].weight).view(z.shape)
        
        #min_encoding_indices.view(z.shape)
        return z_q, min_encoding_indices.view(z.shape[:-1] + (1,))

    def forward(self, z, p=1.0):
        assert z.size(2) == self.e_dim # 256
        zs = torch.split(z, z.size(2) // len(self.embeddings), dim=-1) # B,Td, 128//16 = 8
        zq_i = [self.forward_one(z, i, self.get_hist(i)) for i, z in enumerate(zs)]
        # zq_i contains the quantized latent vectors and the indices of the closest centroids
        # zq_i [z_q, min_encoding_indices]
        # zq_i is length 16, each element is a tuple of z_q and min_encoding_indices
        # z_q is the quantized latent vector, is in dimension B,Td,8
        # min_encoding_indices is the index of the closest centroid, is in dimension B,Td,1

        z_q, min_encoding_indices = [torch.cat([e[i] for e in zq_i], dim=-1) for i in [0,1]]
        # z_q is the concatenation of the quantized latent vectors from each codebook, B,Td,128
        # min_encoding_indices is the concatenation of the indices of the closest centroids from each codebook, B,Td,16

        # compute loss for embedding
        loss = F.mse_loss(z_q,z.detach()) + self.beta * F.mse_loss(z,z_q.detach())
        # the first term if pushing the latent vector to the closest centroid
        # the second term is pushing the centroid to the latent vector

        if p != 1.0:
            # Proba of being quantized.
            quant_mask = torch.bernoulli(p * torch.ones_like(z)).float()
            z_q = quant_mask * z_q + (1 - quant_mask) * z

        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, eos_mask=None):
        """
        Args:
            - indices: [batch_size,seq_len]
        Return:
            - z_q: [batch_size,seq_len,e_dim]
        """
      
        # This is a hack, but it enables us to keep the '-1' index solely in the gpt
        embds = [self.embeddings[str(i)](e.squeeze(-1)) for i, e in enumerate(torch.split(indices, 1, dim=-1))]
        return torch.cat(embds, dim=-1)
    