import torch
import torch.nn as nn
import torch.nn.functional as F
from motion_representation.modules.transformer_vqvae import TransformerLayer
from einops import rearrange
import math

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.vocab_size = config.n_e if hasattr(config,'n_e') else 512 # the whole vocabulary size ( not a single book entries...)
        self.n_books = config.n_books if hasattr(config,'n_books') else 8
        self.n_actions = config.n_actions if hasattr(config,'n_actions') else 8
        self.tok_dim = config.tok_dim if hasattr(config,'tok_dim') else 256 # token dimensions, (a token means multiple books embedding concatenated)
        self.dropout = config.dropout if hasattr(config,'dropout') else 0.1
        self.n_layers = config.n_layers if hasattr(config,'n_layers') else 2 # define number of transformer layers
        self.e_dim = self.tok_dim // self.n_books # as we need concatnate the embedding later, embedding dimensions
        assert self.tok_dim % self.n_books == 0, 'token dimensions must be divisible by n_books'
        self.block_size = config.block_size if hasattr(config,'block_size') else 120 # the length of the sequence
        self.n_heads = config.n_heads if hasattr(config,'n_heads') else 4 # number of heads in the transformer
        self.dataset_hz = 50
        self.max_sample_len = config.max_sample_len * self.dataset_hz if hasattr(config,'max_sample_len') else 10 * self.dataset_hz
        self.pred_idx_classes = self.vocab_size // self.n_books + 1
        self.pred_len = config.pred_len if hasattr(config,'pred_len') else 8
        self.enable_motion_source = config.enable_motion_source
        self.n_sources = config.n_sources
        # define the embedding layers
        self.idx_emb = nn.Embedding(self.vocab_size + self.n_books, self.e_dim)
        self.set_pos_emb()
        self.act_emb = nn.Embedding(self.n_actions, self.tok_dim)
        self.dur_emb = nn.Embedding(self.max_sample_len , self.tok_dim)
        if self.n_sources is not None:
            self.src_emb = nn.Embedding(self.n_sources, self.tok_dim)
            self.fc_emb = nn.Linear(self.tok_dim*5, self.tok_dim) # one idx, one action, one pos
        else:
            self.fc_emb = nn.Linear(self.tok_dim*4, self.tok_dim)


        # define nn
        self.drop = nn.Dropout(self.dropout)
        self.block = nn.Sequential(*[TransformerLayer(self.tok_dim, self.n_heads, self.block_size, self.dropout) for _ in range(self.n_layers)])
        self.ln = nn.LayerNorm(self.tok_dim)

        # define auto-regressive head
        if hasattr(config,'autoreg_head') and config.autoreg_head:
            print("Using autoregressive head")
            self.set_auto_reg_head()
        else:
            print("Not using autoregressive head")
            self.autoreg_head = None
        self.set_head_list()

        self.apply(self._init_weights)

    def set_head_list(self):
        self.head_list = nn.ModuleList()
        for _ in range(self.n_books):
            nb_out = self.pred_idx_classes
            self.head_list.append(nn.Sequential(
                nn.Linear(self.tok_dim, self.tok_dim),
                nn.ReLU(),
                nn.Linear(self.tok_dim, nb_out))
            )

            
    def set_auto_reg_head(self):
        """ An autoregressive head models each index in product quantization autoregressively
        on the previous ones. """
        out_f, n_cb = self.pred_idx_classes, self.n_books
        
        get_head = lambda i: nn.Sequential(nn.Linear(in_features=i * (self.tok_dim // n_cb) +  (n_cb -  i)*out_f, out_features=out_f),
                                                   nn.ReLU(), nn.Linear(out_f, out_f))

        self.autoreg_head = nn.ModuleDict({str(i): get_head(i) for i in range(1, n_cb)})
        


    def set_pos_emb(self):
            n_seqlens, gpt_nembd = self.max_sample_len, self.tok_dim
            pe = torch.zeros(n_seqlens, gpt_nembd)
            position = torch.arange(0, n_seqlens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, gpt_nembd, 2).float() * (-math.log(10000.0) / gpt_nembd))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_emb = nn.Parameter(pe.unsqueeze(0))
            self.pos_emb.requires_grad = False

    def cat_indices(self, idx):
        # input: idx from different books, in range of [0, self.vocab_size/n_books + 1]) as the stop token is included
        # output: concatenated embedding of the idx, in range of [0, self.vocab_size+n_books]
        idx = torch.cat([x + (i *self.pred_idx_classes)# every book has a different range,  
                        for i, x in enumerate(idx.split(1, dim=-1))], dim=-1)
        return idx
    
    def pad(self,z):
       
        b,_,c = z.shape
        pad = torch.zeros((b, 1, c)).long().to(z.device)
        
        return torch.cat([pad,z],dim=1)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def autoreg_along_time(self, x):
        x = self.drop(x)
        x = self.block(x)
        x = self.ln(x)
        logits_t = torch.stack([head(x) for head in self.head_list], dim=2)
        return logits_t

    
    def autoreg_along_head(self, logits_t, idx):
        # input: logits_t[i:], tok_emb[i:]
        # output: logits_idx_t[i+1:]
        # sample the first idx  
        idx_emb = self.idx_emb(idx)
        
        idx_emb = rearrange(idx_emb, 'bs t K c -> (bs t) K c')
        logits_t = rearrange(logits_t, 'bs t K n_e_i -> (bs t) K n_e_i')
        # Split tensors
        idx_emb_split = idx_emb.split(1, dim=-2)  # idx_emb_split: list of empty tensors
        logits_t_split = logits_t.split(1, dim=-2)  # logits_t_split: list of tensors of shape (1, 8, 65)

        out_logits = [logits_t_split[0]]
        for i in range(1,self.n_books):
            in_emb = torch.cat([*idx_emb_split[:i], *logits_t_split[i:]], dim=-1)
            logits = self.autoreg_head[str(i)](in_emb)
            out_logits.append(logits)
        output = rearrange(torch.cat(out_logits, dim=-2), '(bs t) K n_e_i -> bs t K n_e_i', t = idx.shape[1])
        return output


    def get_dummy_idx_emb(self, batch_size, n_codebook, idx):
        # Create a dummy tensor with 1 as temporal dimension to get the correct dimensions for token embedding.
        token_embeddings = self.idx_emb(torch.ones((batch_size * n_codebook), requires_grad=False).to(idx.device).long())
        token_embeddings = token_embeddings.reshape(batch_size, 1, n_codebook, -1)
        token_embeddings = token_embeddings.flatten(2)[:, :0, :]
        return token_embeddings
    
    def forward(self, idx, action, source = None, duration = None, sample = False):
        # output: logits of the next idx [batch_size, T, n_books, codebook_size]
        batch_size, seq_len, n_codebooks = idx.size()
        
        og_idx= idx.clone()
        tem_idx = idx.clone()
        tem_idx = self.cat_indices(tem_idx)
        
        if seq_len != 0:
            idx_embeddings = self.idx_emb(tem_idx.flatten(1))
            idx_embeddings = idx_embeddings.reshape(batch_size, seq_len, n_codebooks, -1)
            idx_embeddings = idx_embeddings.flatten(2)
        else:
            # create a dummy tensor
            idx_embeddings = self.get_dummy_idx_emb(batch_size, n_codebooks, idx)
        
        # pad the idx_embeddings
        # put zeros in the beginning of the sequence
        padded_idx_emb = self.pad(idx_embeddings)
        
        t = padded_idx_emb.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:,:t, :]
        position_embeddings = position_embeddings.repeat(batch_size,1,1)
        act_embeddings = self.act_emb(action)
        act_embeddings = act_embeddings.unsqueeze(1) # B, 1, 256
        act_embeddings = act_embeddings.repeat(1,t,1) # B, T, 256

        dur_embeddings = self.dur_emb(duration)
        dur_embeddings = dur_embeddings.unsqueeze(1)
        dur_embeddings = dur_embeddings.repeat(1,t,1)

        if self.enable_motion_source and source is not None:
            src_embeddings = self.src_emb(source)
            src_embeddings = src_embeddings.unsqueeze(1)
            src_embeddings = src_embeddings.repeat(1,t,1)
            list_emb = [padded_idx_emb, position_embeddings, dur_embeddings,act_embeddings, src_embeddings]
        else: 
            list_emb = [padded_idx_emb, position_embeddings, dur_embeddings,act_embeddings]
                
        # import pdb; pdb.set_trace()
        x = self.fc_emb(torch.cat(list_emb, dim=-1))

        logits_t = self.autoreg_along_time(x) # p(t|t-1, t-2, ... t-n)
        if not sample:
            logits_t = logits_t[:,:-1,:,:] # remove the last token as it is not needed
        # generate logits along head dimension with MLP given the true idx as evidence
        if self.autoreg_head is not None and sample == False:
            logits_idx_t = self.autoreg_along_head(logits_t, tem_idx)
        else:
            logits_idx_t = logits_t
        if not sample:
            loss = self.compute_loss(og_idx, logits_idx_t)
            return loss, logits_idx_t
        else:
            return logits_idx_t

    def compute_loss(self, og_idx,logits):
        # idx in shape B, T, n_books
        # logits in shape B, T, n_books, C
        logits = rearrange(logits, 'bs t K n_e_i -> (bs t) K n_e_i')
        logits = logits.transpose(-1,-2)
        og_idx = rearrange(og_idx, 'bs t K -> (bs t) K')
        
        loss = F.cross_entropy(logits, og_idx)
        return loss
    
    def sample(self,action, source):
        # sample from the logits.
        # output: sampled idx [batch_size, T, n_books]
        # generate logits along time dimension with transformer
        T = self.pred_len*self.dataset_hz //2
        duration = self.pred_len*self.dataset_hz
        # should be half of the actual length as there is a upsample conv in the decoder
        idx = torch.zeros(action.shape[0],0,self.n_books).to(action.device).long()
        duration = torch.tensor(duration).to(action.device).reshape(1)
        for t in range(T):
            idx = idx[:, -self.block_size:,:]
            logits_idx_t = self.forward(idx,action, source, duration=duration, sample = True)[:,-1,:,:] # idx is None at the beginning p(t= 1 | action)
            # [p(i_t^1 | <=t, <i, Q), p(i_t^2 | <=t, <i, Q), ... p(i_t^n | <=t, <i, Q)]
            if self.autoreg_head is not None:
                sampled_idx = self.sample_autoreg_head(logits_idx_t) # sample n_books idx from the logits p(t = 1, idx_1 | action) # [batch_size, n_books, 1]
            else:
                sampled_idx = self.sample_from_logits(logits_idx_t.squeeze(1)) # sample n_books idx from the logits p(t = 1, idx_1 | action) # [batch_size, n_books, 1]
            
            idx = torch.cat([idx, sampled_idx], dim=1)

        return idx

            

    
    def sample_autoreg_head(self, logits):
        # sample idx from the logits
        idx = self.sample_from_logits(logits[:,0,:].unsqueeze(1)) # [batch_size, T= 0, n_books, n_e_i] the idx is in range of [0, self.vocab_size/n_books]
        idx_embd = []
        logits = logits.split(1, dim=-2)
        for i in range(1,self.n_books):
            last_idx_emb = self.idx_emb(self.cat_indices(idx))[:,:,-1,:] # B, 1, c
            idx_embd.append(rearrange(last_idx_emb.unsqueeze(2), 'bs t K c -> (bs t) K c'))
            autoreg_inputs = torch.cat([*idx_embd[:i], *logits[i:]], dim=-1)
            logits_idx = self.autoreg_head[str(i)](autoreg_inputs)
            sampled_idx = self.sample_from_logits(logits_idx)
            idx = torch.cat([idx, sampled_idx], dim=-1)
        return idx

    
    def sample_from_logits(self, logits):
        # sample from the logits
        # output: sampled idx [batch_size, T, n_books]
        K = logits.shape[1]
        logits = rearrange(logits, 'bs K n_e_i -> (bs K) n_e_i')
        prob = F.softmax(logits, dim=-1)
        prob = prob.squeeze(1)
        iz = torch.multinomial(prob, num_samples=1)
        iz = rearrange(iz.squeeze(1), '(bs K) -> bs 1 K', K = K)
        return iz