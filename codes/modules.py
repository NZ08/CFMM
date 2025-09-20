
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from utility.cbn import CBN

class Diffusion_small(nn.Module):
    def __init__(self, noise_steps=100, beta_start=1e-6, beta_end=0.0002, img_size=64, device="cuda"): #yuan beta_start=1e-4 beta_end = 0.02
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x,t): #yuan meiyou h_small
        sqrt_alpha_hat = self.alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = 1 - self.alpha_hat[t][:, None, None, None]
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x, requires_grad=False)
        # Ɛ = h_small
        return sqrt_alpha_hat * x+sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, modelU,n): #yuan n
        modelU.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            # print(reversed(range(1, self.noise_steps)))
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = modelU(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / alpha * (x - ((1 - alpha) / (1 - alpha_hat)) * predicted_noise) + beta * noise
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        modelU.train()
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        x = torch.squeeze(x)
        return x

    def sample_eval(self, modelU,x,n): #yuan n
        modelU.eval()
        with torch.no_grad():
            # x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            # print(reversed(range(1, self.noise_steps)))
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = modelU(x, t)

        modelU.train()
        predicted_noise = (predicted_noise.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        predicted_noise = torch.squeeze(predicted_noise)
        return predicted_noise

    def forward(self, img_size):
        self.img_size = img_size
        return self

class CrossFusion(nn.Module):
    def __init__(self, in_dim1=64, in_dim2=64, k_dim=1, v_dim=1, num_heads=2, n_item=1):
        super(CrossFusion, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        # self.input1 = nn.Linear(in_dim1,4,bias=False)
        # self.input2 = nn.Linear(in_dim2, 4, bias=False)

        # self.proj_q1 = nn.Linear(4, k_dim * num_heads, bias=False)  # in_dim1
        # self.proj_k2 = nn.Linear(4, k_dim * num_heads, bias=False)  # in_dim2
        # self.proj_v2 = nn.Linear(4, v_dim * num_heads, bias=False)  # in_dim2

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False) #in_dim1
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False) #in_dim2
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False) #in_dim2
        self.proj_o = nn.Linear(v_dim * num_heads, 2)
        self.cbn = CBN(lstm_size=n_item, emb_size=in_dim1, out_size= n_item)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, mask=None):
        # x1 = self.input1(x1)
        # x2 = self.input2(x2)
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        # print(batch_size, seq_len1)

        # q1_0 = self.proj_q1(x1).squeeze()
        # k2_0 = self.proj_k2(x2).squeeze()
        # v2_0 = self.proj_v2(x2).squeeze()

        v2_0, _ = self.cbn(x1.squeeze(), x2.squeeze())
        # print(y1.size(), v2_0.size())
        v2_0 = F.normalize(v2_0, p=2, dim=1).unsqueeze(0)
        # print(y1.size(),v2_0.size())
        y1 = F.normalize(x1.squeeze(), p=2, dim=1).unsqueeze(0)
        y2 = F.normalize(x2.squeeze(), p=2, dim=1).unsqueeze(0)

        q1 = self.proj_q1(y1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(y2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(v2_0).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5
        # print(attn.size())
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)  # yuan
        # print(k2.size())
        # attn = torch.softmax(k2, dim=-1)  #yuan
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        # print(output.size())

        # attcbn, _ = self.cbn(output.squeeze(), attcbn)
        output = self.proj_o(output)

        # attn = torch.matmul(q1, k2) / self.k_dim**0.5
        # attn = torch.softmax(attn, dim=-1)
        # output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.sigmoid(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, in_dim1=64, in_dim2=64, k_dim=1, v_dim=1, num_heads=2):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, 2)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size() # seq_len1 是查询序列的长度，seq_len2 是键（和值）序列的长度。
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim**0.5 # 注意力权重矩阵
        # 通过计算查询和键之间的点积，并对其进行缩放，得到了注意力权重张量 attn。这个张量将用于后续的注意力机制中，以加权求和的方式融合序列2中的信息到序列1的表示中。
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)
        # 将注意力权重矩阵 attn 与值矩阵 v2 相乘
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        # torch.matmul(attn, v2) 执行的是批量的矩阵乘法，对于每个批次和每个头，它将 attn 的 (seq_len1, seq_len2) 矩阵与 v2 的 (seq_len2, v_dim) 矩阵相乘，
        # 得到 (seq_len1, v_dim) 的结果。因此，乘法后的维度是 (batch_size, num_heads, seq_len1, v_dim)。
        output = self.proj_o(output)
        return output

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512) #yuan256,512
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256) #yuan512,256

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        self.diffusion_small = Diffusion_small()

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        channels = torch.tensor(channels, device=self.device)
        t = t.to("cuda")

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # noise = torch.randn_like(x, requires_grad=False)
        # t1 = self.diffusion_small.sample_timesteps(1)
        # t=torch.squeeze(t).numpy()
        for i in reversed(range(1, self.diffusion_small.noise_steps)):
            t2 = (torch.ones(1) * i).long().to(self.device)
            alpha = self.diffusion_small.alpha[t2][:, None, None, None]
            alpha_hat = self.diffusion_small.alpha_hat[t2][:, None, None, None]
            beta = self.diffusion_small.beta[t2][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / alpha * (x - ((1 - alpha) / (1 - alpha_hat)) * torch.squeeze(x)) + beta * noise

        # alpha = self.diffusion_small.alpha[t1][:, None, None, None]
        # alpha_hat = self.diffusion_small.alpha_hat[t1][:, None, None, None]
        # beta = self.diffusion_small.beta[t1][:, None, None, None]
        # # noise = torch.randn_like(x)
        # # print(x4.shape, noise.shape, alpha.shape, beta.shape)
        # x = 1 / alpha * (x - ((1 - alpha) / (1 - alpha_hat)) * torch.squeeze(x)) + beta * noise

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4) #yuanx4
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        # output = torch.squeeze(output)
        return output



class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


# if __name__ == '__main__':
#     # net = UNet(device="cpu")
#     # net = UNet_conditional(num_classes=10, device="cpu")
#     print(sum([p.numel() for p in net.parameters()]))
#     x = torch.randn(3, 3, 64, 64)
#     t = x.new_tensor([500] * x.shape[0]).long()
#     y = x.new_tensor([1] * x.shape[0]).long()
#     print(net(x, t, y).shape)
