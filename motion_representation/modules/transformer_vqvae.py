# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
import torch
from torch import nn
from attention import TransformerLayer
from quantizer import VectorQuantizer
from torch.nn import functional as F
from sample import Masked_conv, Masked_up_conv
from motion_representation.utils.utils import Config


class PositionalEncoding(nn.Module):
    """
    Positional encoding module to provide positional information to the model.

    Parameters:
    dim (int): Dimension of the encoding.
    max_len (int): Maximum length of the sequence.
    type (str): Type of positional encoding ('sine_frozen', 'learned', 'none').

    Methods:
    forward(x, start=0): Adds positional encoding to the input tensor.
    """
    def __init__(self, dim, max_len=1024, type='sine_frozen', *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        
        if 'sine' in type:
            rest = dim % 2
            pe = torch.zeros(max_len, dim + rest)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim + rest, 2).float() * (-math.log(10000.0) / (dim + rest)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:, :dim]
            pe = pe.unsqueeze(0)  # [1, t, d]
            if 'ft' in type:
                self.pe = nn.Parameter(pe)
            elif 'frozen' in type:
                self.register_buffer('pe', pe)
            else:
                raise NameError(f"Unknown positional encoding type: {type}")
        elif type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        elif type == 'none':
            # no positional encoding
            pe = torch.zeros((1, max_len, dim))  # [1, t, d]
            self.register_buffer('pe', pe)
        else:
            raise NameError(f"Unknown positional encoding type: {type}")

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class Block(nn.Module):
    """
    Blocks of TransformerLayer with positional encoding.

    Parameters:
    config (Config): Configuration object with necessary parameters.

    Methods:
    forward(x): Forward pass through the blocks of TransformerLayers.
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.pos_all = config.pos_all
        self.block_size = config.block_size
        self.pos = nn.ModuleList([])
        self.TransformLayer_s = nn.ModuleList([])
        # self.conv = nn.Conv1d(config.n_embd, config.n_embd, self.T,stride=1, padding=(self.T - 1) // 2, bias=True, padding_mode='zeros')
        # self.convT = nn.ConvTranspose1d(config.n_embd, config.n_embd, self.T,stride=1, padding=(self.T - 1) // 2, bias=True, padding_mode='zeros')
        for i in range(config.n_layers):
            if self.pos_all or i == 0:
                self.pos.append(PositionalEncoding(config.n_embd, config.block_size, type='sine_frozen'))
            self.TransformLayer_s.append(TransformerLayer(config.n_embd, config.n_heads, config.block_size, config.attn_dropout))

        self.ln = nn.LayerNorm(config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.apply(self._init_weights)
        self.config = config
        if hasattr(config, 'down_conv') and config.down_conv:
            self.down_conv = Masked_conv(config.n_embd, config.n_embd, masked=True)
        else:
            self.down_conv = None
        if hasattr(config, 'up_conv') and config.up_conv:
            self.up_conv = Masked_up_conv(config.n_embd, config.n_embd)
        else:
            self.up_conv = None

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x=None, mask = None):
        # import ipdb; ipdb.set_trace()
        assert x is not None, "Input tensor is None"
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        # T = x.size()[1]
        # assert T <= self.block_size, "Cannot forward, model block size is exhausted."
        # x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        if self.down_conv is not None:
            x,mask = self.down_conv(x, mask)
    
        
        x = self.dropout(x)
        for i in range(len(self.TransformLayer_s)):
            x = self.pos[i](x) if (i == 0 or self.pos_all) else x
            x = self.TransformLayer_s[i](x, mask)

        if self.up_conv is not None:
            x, mask = self.up_conv(x, mask)
        # x = self.convT(x.permute(0,2,1)).permute(0,2,1)
        x = self.ln(x)
        logits = self.proj(x)
        return logits, mask

class TransformerAutoEncoder(nn.Module):
    """
    Transformer-based Autoencoder with encoder and decoder stacks.

    Parameters:
    encoder_config (Config): Configuration for the encoder stack.
    decoder_config (Config): Configuration for the decoder stack.
    config (Config): General configuration.

    Methods:
    encoder_net(x): Forward pass through the encoder.
    decoder_net(x): Forward pass through the decoder.
    """
    def __init__(self, encoder_config, decoder_config, config):
        super().__init__()
        self.encoder = Block(encoder_config)
        self.decoder = Block(decoder_config)
        self.quant_proj = nn.Linear(config.n_embd, config.e_dim)
        self.post_quant_proj = nn.Linear(config.e_dim, config.n_embd)

    def encoder_net(self, x, mask=None):
        o,m = self.encoder(x, mask)
        o = self.quant_proj(o)
        return o, m

    def decoder_net(self, x, mask=None):
        x = self.post_quant_proj(x)
        o,m = self.decoder(x, mask)
        return o,m

class TransformerVQVAE(TransformerAutoEncoder):
    """
    Transformer-based Vector Quantized Variational Autoencoder (VQVAE).

    Parameters:
    in_dim (int): Input dimension.
    out_dim (int): Output dimension.
    n_layers (int): Number of layers.
    n_embd (int): Embedding dimension.
    n_heads (int): Number of attention heads.
    block_size (int): Block size.
    attn_dropout (float): Dropout rate for attention.
    dropout (float): Dropout rate.
    pos_all (bool): Whether to apply positional encoding to all layers.
    n_e (int): Number of embeddings in the codebook.
    e_dim (int): Embedding dimension in the codebook.
    n_books (int): Number of codebooks.
    balance (bool): Balance flag for the codebook.
    beta (float): Beta parameter for the quantizer.

    Methods:
    forward_encoder(x): Forward pass through the encoder.
    forward_decoder(x): Forward pass through the decoder.
    forward(x): Full forward pass through the VQVAE (encoding, quantization, decoding).
    forward_latents(x): Forward pass through the encoder and quantizer.
    forward_from_indices(indices): Forward pass from quantizer indices through the decoder.
    quantize(z): Quantize the latents and compute quantization loss.
    """
    def __init__(self, config):
        self.in_dim = config.in_dim if hasattr(config, 'in_dim') else 256
        self.out_dim = config.out_dim if hasattr(config, 'out_dim') else 256
        self.n_layers = config.n_layers if hasattr(config, 'n_layers') else 2
        self.n_embd = config.n_embd if hasattr(config, 'n_embd') else 256
        self.n_heads = config.n_heads if hasattr(config, 'n_heads') else 8
        self.block_size = config.block_size if hasattr(config, 'block_size') else 512
        self.attn_dropout = config.attn_dropout if hasattr(config, 'attn_dropout') else 0.1
    
        self.dropout = config.dropout if hasattr(config, 'dropout') else 0.2
        self.pos_all = config.pos_all if hasattr(config, 'pos_all') else 0
        self.n_e = config.n_e if hasattr(config, 'n_e') else 256
        self.e_dim = config.e_dim if hasattr(config, 'e_dim') else 128
        self.n_books = config.n_books if hasattr(config, 'n_books') else 8
        self.balance = config.balance if hasattr(config, 'balance') else False

        self.beta = 0.25
        encoder_config = Config(n_layers=self.n_layers, n_embd=self.n_embd, n_heads=self.n_heads, block_size=self.block_size, attn_dropout=self.attn_dropout, dropout=self.dropout, pos_all=self.pos_all, down_conv=True)
        decoder_config = Config(n_layers=self.n_layers, n_embd=self.n_embd, n_heads=self.n_heads, block_size=self.block_size, attn_dropout=self.attn_dropout, dropout=self.dropout, pos_all=self.pos_all, up_conv=True)
        codebook_config = Config(n_e=self.n_e, e_dim=self.e_dim, beta=self.beta, n_books=self.n_books, balance=self.balance)

        super_config = Config(e_dim=codebook_config.e_dim, n_embd=encoder_config.n_embd)
        super().__init__(encoder_config, decoder_config, super_config)
        self.quantizer = VectorQuantizer(codebook_config)
        self.proj_in = nn.Linear(self.in_dim, self.n_embd)
        self.proj_out = nn.Linear(self.n_embd, self.out_dim)

    def log_cosh_loss(self,y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)
    
    def forward_encoder(self, x, mask=None):
        """
        Forward pass through the encoder network.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Encoded tensor.
        """
        x = self.proj_in(x)
        hid, mask = self.encoder_net(x, mask)
        return hid, mask

    def forward_decoder(self, x, mask=None, return_mask=False):
        """
        Forward pass through the decoder network.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Decoded tensor.
        """
        
        x,mask = self.decoder_net(x, mask)
        x = self.proj_out(x)
        if return_mask:
            return x, mask
        else:
            return x

    def forward(self, x, mask=None, mode='train'):
        """
        Full forward pass through the VQVAE.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Total loss and output tensor.
        """
        l_x, hid_mask = self.forward_encoder(x, mask)
        
        z, loss, indices = self.quantize(l_x, hid_mask)
        
        y = self.forward_decoder(z, hid_mask)
        loss_recon = F.mse_loss(y, x, reduction='mean')

        # loss_recon = F.smooth_l1_loss(y, x, reduction='mean', beta=0.01)
        # loss_recon = F.mse_loss(y, x, reduction='mean')
        if mode == 'train':
            total_loss = loss + loss_recon
        else:
            total_loss = loss_recon

        
        return total_loss, y

    def forward_latents(self, x, mask=None):
        """
        Forward pass through the encoder and quantizer.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Quantized tensor and indices.
        """
        
        x, hid_mask = self.forward_encoder(x, mask)
        z, _, indices = self.quantize(x, hid_mask)
        # remove -1 indices
        indices += 1
        return z, indices, hid_mask

    def forward_from_indices(self, indices):
        """
        Forward pass from quantizer indices through the decoder.

        Parameters:
        indices (Tensor): Quantizer indices.

        Returns:
        Tensor: Output tensor.
        """
        indices -= 1 # recover -1 indices # now the indices contains -1
        # build valid mask based on estimated indices
        valid = (1. - (torch.cumsum(((indices == -1).sum(-1) > 0).float(), dim=-1) > 0).float()).int()
        valid = valid.unsqueeze(-1)
        # need to remove -1 indices
        indices = valid * indices + (1 - valid) * torch.zeros_like(indices)

        # set the -1 indices to 0
        z = self.quantizer.get_codebook_entry(indices)
        y,ext_valid = self.forward_decoder(z, valid, return_mask=True)
        
        return y, ext_valid

    def quantize(self, z, mask = None):
        """
        Quantize the latents and compute quantization loss.

        Parameters:
        z (Tensor): Input latents.

        Returns:
        Tensor: Quantized latents, quantization loss, and indices.
        """

        z_q, loss, indices = self.quantizer(z)
        if mask is not None:
            loss = torch.sum(loss * mask) / torch.sum(mask)
            
            indices = (indices) * mask + (1 - mask) * -1 * torch.ones_like(indices)
            # -1 means that the timestamp is not valid
            z_q = z_q * mask
        return z_q, loss, indices
