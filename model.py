
import torch
import torch.nn as nn
import torch.nn.functional as F

def _safen(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

class DCL_GRU_TCN(nn.Module):
    def __init__(self, channels, num_nodes, input_len, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.input_len = input_len

        self.low_gru = nn.GRU(input_size=1, hidden_size=channels, batch_first=True)

        self.high_tcn = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),  # 第三层
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, low_freq, high_freq):  # [B,1,N,L]
        low_freq = _safen(low_freq)
        high_freq = _safen(high_freq)
        B, _, N, L = low_freq.shape

        low_seq = low_freq.permute(0, 2, 3, 1).reshape(B * N, L, 1)   # (B*N, L, 1)
        low_out, _ = self.low_gru(low_seq)                             # (B*N, L, C)
        H_low = low_out.reshape(B, N, L, self.channels).permute(0, 3, 1, 2)  # [B,C,N,L]

        H_high = self.high_tcn(high_freq)  # [B,C,N,L]

        G = torch.sigmoid(self.gate(torch.cat([H_low, H_high], dim=1)))  # [B,C,N,L]
        H_seq = G * H_high + (1.0 - G) * H_low
        return _safen(H_seq)

class ResGatedTCNBlock(nn.Module):
    def __init__(self, channels, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.norm = nn.GroupNorm(1, channels)
        self.dw = nn.Conv2d(
            channels, 2 * channels, kernel_size=(1, kernel),
            padding=(0, pad), dilation=(1, dilation), groups=channels
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        z = self.norm(x)
        z = self.dw(z)
        a, b = z.chunk(2, dim=1)
        z = a * torch.sigmoid(b)
        z = self.pw(z)
        z = self.dropout(z)
        return x + self.alpha * z

# =========================
class SE2D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):  # [B,C,N,L]
        s = x.mean(dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * w

# =========================
class TCNSERefine(nn.Module):
    def __init__(self, channels, tcn_dilations=(1, 2, 4), tcn_dropout=0.1, se_reduction=4):
        super().__init__()
        self.tcn = nn.Sequential(
            *[ResGatedTCNBlock(channels, kernel=3, dilation=d, dropout=tcn_dropout) for d in tcn_dilations]
        )
        self.alpha_tcn = nn.Parameter(torch.tensor(0.5))  # TCN 残差强度

        self.se = SE2D(channels, reduction=se_reduction)
        self.alpha_se = nn.Parameter(torch.tensor(0.5))   # SE 残差强度

    def forward(self, x):  # [B,C,N,L]
        z_tcn = self.tcn(x)
        z = x + self.alpha_tcn * (z_tcn - x)
        z_se = self.se(z)
        z = z + self.alpha_se * (z_se - z)
        return z


# =========================
class VGTMSN(nn.Module):

    def __init__(self, device, input_dim=1, channels=64, num_nodes=8,
                 input_len=12, output_len=1, dropout=0.1,
                 tcn_dilations=(1, 2, 4), tcn_dropout=0.1, se_reduction=4,
                 **kwargs):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.channels = channels
        self.input_len = input_len
        self.output_len = output_len

        self.smooth_k = 5
        w = torch.ones(1, 1, self.smooth_k) / self.smooth_k
        self.register_buffer("ma_weight", w)

        self.dcl = DCL_GRU_TCN(channels=channels, num_nodes=num_nodes, input_len=input_len, dropout=dropout)

        self.refine = TCNSERefine(
            channels, tcn_dilations=tcn_dilations, tcn_dropout=tcn_dropout, se_reduction=se_reduction
        )

        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, output_len, kernel_size=1),
        )

    def _low_high_split(self, x):  # x:[B,1,N,L]
        B, C, N, L = x.shape
        xv = x.reshape(B * N, 1, L)
        k = int(self.ma_weight.size(-1))
        pad = max(0, min((k - 1) // 2, L - 1))
        w = self.ma_weight.to(dtype=x.dtype)
        xv_pad = F.pad(xv, (pad, pad), mode='reflect') if pad > 0 else xv
        low = F.conv1d(xv_pad, w).reshape(B, 1, N, L)
        high = x - low
        return _safen(low), _safen(high)

    def _prepare_low_high(self, x, do_low, do_high):
        if (do_low is None) or (do_high is None):
            low, high = self._low_high_split(x)
        else:
            low, high = do_low, do_high
        low = _safen(low.to(x.dtype).to(x.device))
        high = _safen(high.to(x.dtype).to(x.device))
        B, C, N, L = low.shape
        if C != 1:
            low, high = low[:, :1], high[:, :1]
        if N == 1 and self.num_nodes > 1:
            low = low.expand(-1, 1, self.num_nodes, -1)
            high = high.expand(-1, 1, self.num_nodes, -1)
        elif N != self.num_nodes:
            raise ValueError(f"do_low/do_high 的节点数 {N} 与 num_nodes={self.num_nodes} 不一致")
        return low, high

    def forward(self, history_data, dataset_index=None, do_low=None, do_high=None):
        x = _safen(history_data)

        low_freq, high_freq = self._prepare_low_high(x, do_low, do_high)
        H_seq = self.dcl(low_freq, high_freq)

        H_seq = self.refine(H_seq)

        y_seq = self.head(H_seq)
        y = y_seq[:, :, :, -1:]
        y = _safen(y)
        return y.permute(0, 3, 2, 1)
