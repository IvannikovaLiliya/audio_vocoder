import torch
import torch.nn.functional as F
import torch.nn as nn
from free_v.dataset import inverse_mel
from vocos.modules import ConvNeXtBlock

LRELU_SLOPE = 0.1

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.ASP_num_kernels = len(h.ASP_resblock_kernel_sizes)

        self.dim = 512
        self.num_layers = 8
        self.adanorm_num_embeddings = None
        self.intermediate_dim = 1536
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        layer_scale_init_value = 1 / self.num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.convnext2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.h.ASP_channel,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                # for _ in range(self.num_layers)
                for _ in range(1)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.final_layer_norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mel, inv_mel=None):
        if inv_mel is None:
            inv_amp = (
                inverse_mel(
                    mel,
                    self.h.n_fft,
                    self.h.num_mels,
                    self.h.sampling_rate,
                    self.h.hop_size,
                    self.h.win_size,
                    self.h.fmin,
                    self.h.fmax,
                )
                .abs()
                .clamp_min(1e-5)
            )
        else:
            inv_amp = inv_mel
        logamp = inv_amp.log()
        # logamp = self.ASP_input_conv(logamp)
        for conv_block in self.convnext2:
            logamp = conv_block(logamp, cond_embedding_id=None)
        # logamp = self.final_layer_norm2(logamp.transpose(1, 2))
        # logamp = logamp.transpose(1, 2)
        # logamp = self.ASP_output_conv(logamp)

        pha = self.PSP_input_conv(mel)
        pha = self.norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        for conv_block in self.convnext:
            pha = conv_block(pha, cond_embedding_id=None)
        pha = self.final_layer_norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)

        pha = torch.atan2(I, R)

        rea = torch.exp(logamp) * torch.cos(pha)
        imag = torch.exp(logamp) * torch.sin(pha)

        spec = torch.complex(rea, imag)
        # spec = torch.cat((rea.unsqueeze(-1), imag.unsqueeze(-1)), -1)

        audio = torch.istft(
            spec,
            self.h.n_fft,
            hop_length=self.h.hop_size,
            win_length=self.h.win_size,
            window=torch.hann_window(self.h.win_size).to(mel.device),
            center=True,
        )

        return logamp, pha, rea, imag, audio.unsqueeze(1)