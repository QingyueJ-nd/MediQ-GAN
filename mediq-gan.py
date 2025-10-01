import os, math, random, argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader

import torchvision
import torchvision.utils as vutils
import torchvision.transforms as T
from torchvision import datasets

import matplotlib.pyplot as plt
import pennylane as qml

from data_utils import CustomImageDataset, scale_data, get_noise_upper_bound

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train','generate','generate_epoch'])
parser.add_argument('--version', type=int, default=0)

parser.add_argument('--input_size', type=int, default=64)
parser.add_argument('--crop_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='ISIC2019', choices=['ISIC2019', 'ODIR-5k', 'RetinaMNIST'])
parser.add_argument('--output_dir', type=str, default='runs/',
                    help='Root directory for checkpoints and generated images')

# Devices
parser.add_argument('--force_cpu_quantum', type=int, default=1, help='Run PQCs on CPU even if CUDA is available')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--noise_size', type=int, default=128)

# Label embedding knobs
parser.add_argument('--label_embedding_dim', type=int, default=128)
parser.add_argument('--gen_label_embedding_dim', type=int, default=32)
parser.add_argument('--disc_embed_dim', type=int, default=128)

# Encoder/Decoder choices
parser.add_argument('--encoder_type', type=str, default='cnn', choices=['cnn', 'fc', 'vanilla'],
                    help='cnn: paper conv encoder; fc: MLP; vanilla: simpler conv + fc encoder')
parser.add_argument('--decoder_type', type=str, default='style2', choices=['style2', 'vanilla'],
                    help='style2: ResUp+blur; vanilla: fc + conv decoder')
parser.add_argument('--skip_connections', type=int, default=1)
parser.add_argument('--skip_inject_scale', type=float, default=0.05,
                    help='Scale for injecting the 4x4 skip into the output')


# Quantum/classical split
parser.add_argument('--n_qubits', type=int, default=16)
parser.add_argument('--q_depth', type=int, default=6)
parser.add_argument('--n_generators', type=int, default=5, help='Set 0 for purely classical generator')
parser.add_argument('--q_delta', type=float, default=1.0)
parser.add_argument('--feature_split_ratio', type=float, default=0.5, help='fraction of 32ch (4x4) routed to quantum')

# Decoder (vanilla) sizing
parser.add_argument('--vanilla_dec_latent_dim', type=int, default=100)
parser.add_argument('--vanilla_dec_init_size', type=int, default=8)
parser.add_argument('--vanilla_dec_init_ch', type=int, default=64)

# Blur/attention knobs (style2)
parser.add_argument('--use_blur', type=int, default=1, help='Use anti-alias blur before convs in up blocks')
parser.add_argument('--use_attn32', type=int, default=1, help='Self-attention at 32x32')

# GAN options
parser.add_argument('--gan_type', type=str, default='wgan-gp', choices=['vanilla', 'wgan-gp'])
parser.add_argument('--lambda_gp', type=float, default=10.0)

# Prototypes
parser.add_argument('--use_proto', type=int, default=0)
parser.add_argument('--proto_path', type=str, default='')
parser.add_argument('--proto_in_dim', type=int, default=1280)
parser.add_argument('--proto_sigma_min', type=float, default=0.2)
parser.add_argument('--proto_sigma_max', type=float, default=0.8)
parser.add_argument('--lambda_proto_reg', type=float, default=5e-3)
parser.add_argument('--proto_reg_margin', type=float, default=1.0,
                    help='Margin for distance-based prototype separation (0 = disabled; hinge off)')
parser.add_argument('--proto_reg_normalize', type=int, default=1,
                    help='Normalize prototypes before computing distances (1=yes, 0=no)')


# LRs
parser.add_argument('--lrG_quantum', type=float, default=3e-1)
parser.add_argument('--lrG_encoder', type=float, default=8e-5)
parser.add_argument('--lrG_decoder', type=float, default=8e-5)
parser.add_argument('--lrD', type=float, default=2.5e-4)

# Seed
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# -----------------------------
# Seeding
# -----------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# -----------------------------
# Data / basic vars
# -----------------------------
input_size = args.input_size
crop_size = args.crop_size
batch_size = args.batch_size
nz = args.noise_size

USE_PROTO = args.use_proto == 1
gen_label_emb_dim = (args.gen_label_embedding_dim if USE_PROTO else args.label_embedding_dim)
disc_embed_dim = args.disc_embed_dim

# default device is CUDA for everything; PQC can be on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pqc_device = torch.device('cpu') if args.force_cpu_quantum else device
print(f"device: {device} | pqc_device: {pqc_device}")

# -----------------------------
# Dataset loading
# -----------------------------
if args.dataset == "ISIC2019":
    root_directory = "ISIC_2019_train/sampled_20pct/"
    if args.proto_path == "":
        args.proto_path = "prototype/isic2019_avg.pt"
elif args.dataset == "ODIR-5k":
    root_directory = "ODIR-5k_Train/labeled_input"
    if args.proto_path == "":
        args.proto_path = "prototype/ODIR-5k_avg.pt"
elif args.dataset == "RetinaMNIST":
    root_directory = "RetinaMNIST/labeled_input"
    if args.proto_path == "":
        args.proto_path = "prototype/RetinaMNIST_avg.pt"

transform = T.Compose([
    T.Resize((input_size, input_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
dataset = datasets.ImageFolder(root=root_directory, transform=transform)
num_classes = len(dataset.classes)
workers = 4 if torch.cuda.is_available() else 2
kwargs = {'num_workers': workers, 'pin_memory': device.type == 'cuda'}
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
print(f"Dataset: {args.dataset}, len(dataloader): {len(dataloader)}, num_classes: {num_classes}")

# -----------------------------
# Inits
# -----------------------------
def _init_one(m, act: str = "lrelu"):
    if isinstance(m, nn.Conv2d):
        nonlin = "leaky_relu" if act == "lrelu" else "relu"
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain("leaky_relu", 0.2) if act == "lrelu" else np.sqrt(2.0)
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None: nn.init.constant_(m.bias, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
        if hasattr(m, "weight") and m.weight is not None: nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

# -----------------------------
# Utility blocks
# -----------------------------
class Blur(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]]) / 16.0
        self.register_buffer('weight', k[None, None, :, :].repeat(channels,1,1,1))
        self.groups = channels
        self.pad = nn.ReflectionPad2d(1)
    def forward(self, x):
        x = self.pad(x)
        return F.conv2d(x, self.weight, stride=1, padding=0, groups=self.groups)

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.q = nn.Conv2d(ch, ch//8, 1)
        self.k = nn.Conv2d(ch, ch//8, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B,C,H,W = x.shape
        q = self.q(x).view(B, -1, H*W).transpose(1,2)   # B,N,Cq
        k = self.k(x).view(B, -1, H*W)                  # B,Ck,N
        attn = torch.bmm(q, k) / math.sqrt(k.size(1))   # B,N,N
        attn = F.softmax(attn, dim=-1)
        v = self.v(x).view(B, C, H*W)                   # B,C,N
        out = torch.bmm(v, attn.transpose(1,2)).view(B, C, H, W)
        return x + self.gamma * out

class ResUp(nn.Module):
    def __init__(self, in_ch, out_ch, use_blur=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.blur = Blur(in_ch) if use_blur else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        s = self.up(x)
        s = self.skip(s)
        x = self.up(x)
        x = self.blur(x)
        x = F.leaky_relu(self.in1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.in2(self.conv2(x)), 0.2, inplace=True)
        return x + s

# -----------------------------
# Critic / Discriminator
# -----------------------------
class Critic(nn.Module):
    def __init__(self, nc=3, ndf=64, num_classes=8, embed_dim=128, use_cosine=False, init_scale=1.0):
        super().__init__()
        self.use_cosine = use_cosine
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.trunk = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),   # 64->32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),# 32->16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),#16->8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),#8->4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ndf*8, 1)
        self.feat_proj = nn.Linear(ndf*8, embed_dim, bias=False)
        if self.use_cosine:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
    def forward(self, x, y):
        h = self.trunk(x)
        h_gap = self.gap(h).view(h.size(0), -1)
        out = self.linear(h_gap)
        fproj = self.feat_proj(h_gap)
        emb = self.embed(y)
        if self.use_cosine:
            fproj = F.normalize(fproj, dim=1)
            emb = F.normalize(emb, dim=1)
            proj = self.scale * torch.sum(fproj * emb, dim=1, keepdim=True)
        else:
            proj = torch.sum(fproj * emb, dim=1, keepdim=True)
        return (out + proj).view(-1)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, num_classes=8, label_embedding_dim=10):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.label_projection = nn.Linear(label_embedding_dim, input_size * input_size)
        self.model = nn.Sequential(
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input, labels):
        label_embedding = self.label_embedding(labels)
        label_channel = self.label_projection(label_embedding).view(-1,1,input_size,input_size)
        x = torch.cat([input, label_channel], dim=1)
        return self.model(x).view(-1)

# -----------------------------
# Encoders
# -----------------------------
class FCEncoder(nn.Module):
    def __init__(self, in_dim, classical_latent_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, classical_latent_dim)
        )
        self.apply(lambda m: _init_one(m, "lrelu"))
    def forward(self, zc):
        return self.features(zc)

class CNNFeatureEncoder(nn.Module):
    def __init__(self, in_dim, out_ch=32):
        super().__init__()
        self.in_dim = in_dim
        self.out_ch = out_ch
        self.up8 = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 128, 1, 1, 0), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1),   nn.GELU(),
            nn.Conv2d(128, out_ch, 3, 1, 1),nn.GELU(),
        )
        self.apply(lambda m: _init_one(m, "lrelu"))

    def forward(self, zc):
        B = zc.size(0)
        x = zc.view(B, self.in_dim, 1, 1)
        x = self.up8(x)
        fmap = self.conv(x)
        return fmap


class VanillaEncoder(nn.Module):
    def __init__(self, input_channels=128, output_dim=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2)
        )
        self.apply(lambda m: _init_one(m, "lrelu"))
    def forward(self, x):
        return self.features(x)

class SimpleUp(nn.Module):
    def __init__(self, in_ch, out_ch, use_blur=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.blur = Blur(in_ch) if use_blur else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        s = self.up(x); s = self.skip(s)
        x = self.up(x); x = self.blur(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + s

class NoiseInject(nn.Module):
    """Adds per-channel, learned-scaled Gaussian noise."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        if x.size(0) == 0:
            return x
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class ClassicalDecoder(nn.Module):
    def __init__(self, in_channels=32, out_channels=3, img_size=64,
                 init_size=8, init_channels=64):
        super().__init__()
        self.img_size = img_size
        self.init_size = init_size
        self.init_channels = init_channels
        self.feature_dim = self.init_channels * self.init_size * self.init_size

        self.fc_to_map = nn.Sequential(
            nn.Linear(in_channels * self.init_size * self.init_size, self.feature_dim),
            nn.ReLU(True),
            nn.Unflatten(1, (self.init_channels, self.init_size, self.init_size))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.apply(lambda m: _init_one(m, "lrelu"))

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.fc_to_map(x)
        x = self.decoder(x)
        return x

# -----------------------------
# Prototypes
# -----------------------------
class ProtoReducer(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes, sigma_min=0.05, sigma_max=0.5):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.class_log_scales = nn.Parameter(torch.zeros(num_classes))
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim))
        self.sigma_min = float(sigma_min); self.sigma_max = float(sigma_max)
    def forward(self, labels):
        mu_in = self.prototypes[labels]
        z_mean = self.reducer(mu_in)
        sigma = F.softplus(self.class_log_scales[labels])
        sigma = sigma.clamp(self.sigma_min, self.sigma_max).unsqueeze(1).expand_as(z_mean)
        return z_mean, sigma

# -----------------------------
# Quantum setup
# -----------------------------
USE_QUANTUM = (args.n_generators > 0)
if USE_QUANTUM:
    assert args.n_qubits == 16, "Patch-wise tokens are 4x4 -> require n_qubits=16"
    dev = qml.device("lightning.qubit", wires=args.n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def quantum_circuit(noise, weights):
        # per-qubit input encoding
        for i in range(args.n_qubits):
            qml.RY(noise[i], wires=i)
            qml.RX(noise[i], wires=i)
        # layers
        for i in range(args.q_depth):
            for y in range(args.n_qubits):
                qml.RY(weights[i][y], wires=y)
            for y in range(args.n_qubits-1):
                qml.CZ(wires=[y, y+1])
        return [qml.expval(qml.PauliX(i)) for i in range(args.n_qubits)]

def run_pqc_batch(tokens, q_params):
    """
    tokens: [B, k, 16]  (CPU or GPU)
    q_params: list of [q_depth, 16] tensors (on pqc_device)
    Returns: [B, k, 16] on tokens.device
    """
    if tokens.numel() == 0:
        return tokens
    B, K, Q = tokens.shape
    out = torch.zeros(B, K, Q, dtype=tokens.dtype, device=tokens.device)
    cpu_tokens = tokens.to(pqc_device)
    for g_idx, theta in enumerate(q_params):
        theta_cpu = theta.to(pqc_device)
        rows = []
        for b in range(B):
            f = quantum_circuit(cpu_tokens[b, g_idx], theta_cpu)
            rows.append(torch.stack(f) if isinstance(f, (list, tuple)) else f)
        block = torch.stack(rows, dim=0).to(tokens.device)
        out[:, g_idx] = block
    return out

# -----------------------------
# Hybrid Generator
# -----------------------------
class HybridGenerator(nn.Module):
    def __init__(self, num_classes=8, label_embedding_dim=32, img_size=64,
                 cls_ch=16, q_ch=16, n_generators=5, encoder_type='cnn', decoder_type='style2'):
        super().__init__()
        self.k = n_generators
        self.img_size = img_size
        self.encoder_type = encoder_type
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.label_norm = nn.LayerNorm(label_embedding_dim)

        # ---------------------
        # Conditioning
        # ---------------------
        self.in_dim = nz + label_embedding_dim  # (proto sampling replaces z draw, not its dimensionality)

        # ---------------------
        # Encoder
        # ---------------------
        if self.encoder_type == 'cnn':
            self.encoder = CNNFeatureEncoder(in_dim=self.in_dim, out_ch=32)
        else:
            self.encoder_fc = nn.Sequential(
                nn.Linear(self.in_dim, 32 * 8 * 8)
            )
            nn.init.orthogonal_(self.encoder_fc[0].weight, gain=np.sqrt(2.0))
            nn.init.constant_(self.encoder_fc[0].bias, 0.02)
            self.enc_refine = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1), nn.GELU(),
                nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(),
            )

    
        split_q = int(round(32 * args.feature_split_ratio))
        split_q = max(1, min(31, split_q))
        self.q_in_ch = split_q
        self.cls_in_ch = 32 - split_q

        self.proj_cls_in = nn.Conv2d(32, self.cls_in_ch, 1)
        if self.k > 0:
            self.proj_q_in = nn.Conv2d(32, self.q_in_ch, 1)

            # PQC input aligner: per-site vector (len=q_in_ch) -> n_qubits
            self.q_align = nn.Linear(self.q_in_ch, args.n_qubits)

            # PQC parameters
            self.q_params = nn.ParameterList([
                nn.Parameter(args.q_delta * torch.rand(args.q_depth, args.n_qubits), requires_grad=True)
                for _ in range(self.k)
            ])

            # Expand PQC outputs (k scalar maps at 4x4) to q_ch channels
            self.proj_qexp = nn.Conv2d(self.k, q_ch, 1)
        else:
            self.q_align = None
            self.q_params = nn.ParameterList([])
            self.proj_qexp = None

        self.classical_branch = nn.Sequential(
            nn.Conv2d(self.cls_in_ch, self.cls_in_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ---------------------
        # Decoders
        # ---------------------
        use_blur = bool(args.use_blur)
        fusion_ch = self.cls_in_ch + (q_ch if self.k > 0 else 0)
        self.fusion_ch = fusion_ch  # keep for adapters

        if args.decoder_type == 'style2':
            self.decoder_style2 = nn.Sequential(
                
                SimpleUp(self.fusion_ch, 128, use_blur=False),
                NoiseInject(128),
                nn.LeakyReLU(0.2, inplace=True),

                
                SimpleUp(128, 96, use_blur=False),
                NoiseInject(96),
                nn.LeakyReLU(0.2, inplace=True),

                
                SimpleUp(96, 64, use_blur=False),
                NoiseInject(64),
                nn.LeakyReLU(0.2, inplace=True),

            
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Tanh(),
            )
            self.decoder = self.decoder_style2


        else:
            self.decoder_vanilla = ClassicalDecoder(
                in_channels=fusion_ch, out_channels=3, img_size=64,
                init_size=8, init_channels=64
            )
        
            self.decoder = self.decoder_vanilla

        self.apply(lambda m: _init_one(m, "lrelu"))

    # ---------------------
    # Helpers
    # ---------------------
    def _condition(self, z, labels):
        emb = self.label_norm(self.label_embedding(labels))
        return torch.cat([z, emb], dim=1)  # z_c

    def _encode_fmap(self, zc):
        if self.encoder_type == 'cnn':
            return self.encoder(zc)
        else:
            B = zc.size(0)
            x = self.encoder_fc(zc).view(B, 32, 8, 8)
            x = self.enc_refine(x)
            return x

    def _quantum_branch_8to4(self, F_q_in_8x8):
        B = F_q_in_8x8.size(0)

        
        F_q_4x4 = F.avg_pool2d(F_q_in_8x8, kernel_size=2, stride=2)


        sites = F_q_4x4.view(B, self.q_in_ch, 16).transpose(1, 2)
        aligned = self.q_align(sites)

        # Select k site indices (round-robin over 16 slots)
        k = self.k
        site_idx = torch.linspace(0, 15, steps=k).round().long().tolist() if k > 0 else []
        tokens = []
        for si in site_idx:
            tokens.append(aligned[:, si, :])                           # [B, n_qubits]
        if k == 0:
            return None

        Tok = torch.stack(tokens, dim=1)                               # [B, k, n_qubits]
        q_out = run_pqc_batch(Tok, self.q_params)                      # [B, k, n_qubits]

        # Reduce qubit dimension to a scalar per token/site (mean over qubits)
        scalars = q_out.mean(dim=2)                                    # [B, k]

        # Place back onto a 4x4 grid: one site per generator index
        q_maps_4x4 = torch.zeros(B, k, 4, 4, device=F_q_in_8x8.device, dtype=F_q_in_8x8.dtype)
        for gi, si in enumerate(site_idx):
            y, x = divmod(int(si), 4)
            q_maps_4x4[:, gi, y, x] = scalars[:, gi]

        # Project to q_ch channels and upsample to 8x8 for fusion
        q_ch_maps_4x4 = self.proj_qexp(q_maps_4x4)                    # [B, q_ch, 4, 4]
        q_ch_maps_8x8 = F.interpolate(q_ch_maps_4x4, scale_factor=2, mode='bilinear', align_corners=False)

        return q_ch_maps_8x8

    # ---------------------
    # Forward
    # ---------------------
    def forward(self, z, labels):
        # Condition
        zc = self._condition(z, labels)


        fmap = self._encode_fmap(zc)

        
        F_cls_in = self.proj_cls_in(fmap)
        F_cls = self.classical_branch(F_cls_in)

        if self.k > 0:
            F_q_in = self.proj_q_in(fmap)
            F_q = self._quantum_branch_8to4(F_q_in)
            F_fuse = torch.cat([F_cls, F_q], dim=1)
        else:
            F_fuse = F_cls

        if args.decoder_type == 'vanilla':
            img = self.decoder(F_fuse)
        else:
            img = self.decoder(F_fuse)

        return img

# -----------------------------
# WGAN-GP utils
# -----------------------------
def compute_gradient_penalty(critic, real_data, fake_data, labels):
    B = real_data.size(0)
    epsilon = torch.rand(B, 1, 1, 1, device=real_data.device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)
    critic_interpolated = critic(interpolated, labels)
    gradients = torch_grad(outputs=critic_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones_like(critic_interpolated),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(B, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

# -----------------------------
# Prototype regularize
# -----------------------------
def prototype_regularizer(generator, lambda_proto_reg: float, margin: float = 0.0, normalize: bool = True):
    """
    Returns a scalar regularization term to ADD to g_loss.
    - If margin > 0: hinge on pairwise distances: mean(ReLU(margin - d_ij))
      (penalizes pairs that are too close)
    - If margin == 0: maximize mean distance by returning the NEGATED mean distance
      (so adding it reduces the loss as distances grow)
    """
    if not hasattr(generator, "proto"):
        return None
    if lambda_proto_reg <= 0:
        return None

    proto = generator.proto.prototypes  # [K, D]
    if normalize:
        proto = F.normalize(proto, dim=1)

    D = torch.cdist(proto, proto, p=2)        # [K, K]
    K = D.size(0)
    mask = ~torch.eye(K, dtype=torch.bool, device=D.device)

    if margin > 0.0:
        # Penalize distances smaller than margin
        reg = F.relu(margin - D[mask]).mean()
        return lambda_proto_reg * reg
    else:
        # Encourage large distances by subtracting their mean
        mean_dist = D[mask].mean()
        return (-lambda_proto_reg) * mean_dist


# -----------------------------
# Checkpoint manager
# -----------------------------
class CheckpointManager:
    def __init__(self, dataset, version):
        self.base_dir = os.path.join(args.output_dir, dataset, f'v{version}', 'ckpt')
        os.makedirs(self.base_dir, exist_ok=True)
    def save_checkpoint(self, netG, netD, optimG, optimD, epoch, loss_G, loss_D, additional_info=None):
        simple_path = os.path.join(self.base_dir, f'checkpoint_epoch_{epoch:03d}.pth')
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizer_G_state_dict': optimG.state_dict(),
            'optimizer_D_state_dict': optimD.state_dict(),
            'loss_G': loss_G,
            'loss_D': loss_D,
            'additional_info': additional_info
        }
        torch.save(checkpoint, simple_path)
        latest_path = os.path.join(self.base_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Checkpoint saved: {simple_path} and updated latest_checkpoint.pth")
    def load_checkpoint(self, netG, netD, optimG, optimD, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.base_dir, 'latest_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}.")
            return None
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        netG.load_state_dict(checkpoint['generator_state_dict'])
        netD.load_state_dict(checkpoint['discriminator_state_dict'])
        optimG.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimD.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

# -----------------------------
# Builders
# -----------------------------
def _build_models():
    generator = HybridGenerator(
        num_classes=num_classes,
        label_embedding_dim=gen_label_emb_dim,
        img_size=input_size,
        cls_ch=16, q_ch=16,
        n_generators=args.n_generators,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type
    ).to(device)

    if USE_PROTO:
        print("Using learnable class prototypes with warm-start.")
        assert args.proto_path != "", "Pass --proto_path to warm-start prototypes."
        proto_dict = torch.load(args.proto_path, map_location='cpu')
        assert len(proto_dict) == num_classes, f"proto_dict has {len(proto_dict)} classes, dataset has {num_classes}"
        prototype_matrix = torch.stack([proto_dict[c] for c in range(num_classes)])
        generator.proto = ProtoReducer(
            in_dim=args.proto_in_dim,
            out_dim=nz,
            num_classes=num_classes,
            sigma_min=args.proto_sigma_min,
            sigma_max=args.proto_sigma_max
        ).to(device)
        with torch.no_grad():
            generator.proto.prototypes.copy_(prototype_matrix.to(device))

    if args.gan_type == 'wgan-gp':
        discriminator = Critic(nc=3, ndf=64, num_classes=num_classes, embed_dim=disc_embed_dim).to(device)
    else:
        discriminator = Discriminator(nc=3, ndf=64, num_classes=num_classes,
                                      label_embedding_dim=args.label_embedding_dim).to(device)

    # group params
    g_param_groups = []
    enc_like = []
    dec_params = list(generator.decoder.parameters()); dec_ids = set(map(id, dec_params))
    proto_params = list(generator.proto.parameters()) if USE_PROTO and hasattr(generator, 'proto') else []
    proto_ids = set(map(id, proto_params))
    q_params   = list(generator.q_params) if hasattr(generator, 'q_params') else []
    q_ids      = set(map(id, q_params))

    for name, p in generator.named_parameters():
        pid = id(p)
        if pid in dec_ids or pid in q_ids or pid in proto_ids: 
            continue
        enc_like.append(p)

    if enc_like:
        g_param_groups.append({'params': enc_like, 'lr': args.lrG_encoder})
    if dec_params:
        g_param_groups.append({'params': dec_params, 'lr': args.lrG_decoder})
    if q_params:
        g_param_groups.append({'params': q_params, 'lr': args.lrG_quantum})
    if USE_PROTO and proto_params:
        g_param_groups.append({'params': proto_params, 'lr': args.lrG_encoder})

    optim_G = optim.Adam(g_param_groups, betas=(0.5, 0.999))
    if args.gan_type == 'wgan-gp':
        optim_D = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(0.0, 0.9))
    else:
        optim_D = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(0.5, 0.999))
    return generator, discriminator, optim_G, optim_D

def _build_models_for_infer():
    return _build_models()

# -----------------------------
# Training
# -----------------------------
def train_qgan(num_epochs=50, n_critic=5):
    generator, discriminator, optim_G, optim_D = _build_models()

    if args.gan_type == 'vanilla':
        criterion = nn.BCELoss()
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

    gen_losses, disc_losses = [], []
    ckpt = CheckpointManager(args.dataset, args.version)
    start_epoch = 0
    last = ckpt.load_checkpoint(generator, discriminator, optim_G, optim_D)
    if last is not None:
        start_epoch = last['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

    noise_upper_bound = math.pi/8
    original_ratio = None
    upper_bounds = [noise_upper_bound]

    for epoch in range(start_epoch, start_epoch + num_epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            real_imgs = real_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if args.gan_type == 'wgan-gp':
                for _ in range(n_critic):
                    optim_D.zero_grad()
                    real_validity = discriminator(real_imgs, labels)
                    d_real = -real_validity.mean()

                    if USE_PROTO:
                        mu_z, sigma = generator.proto(labels)
                        eps = torch.randn_like(mu_z)
                        noise = mu_z + sigma * eps
                    else:
                        noise = torch.randn(batch_size, nz, device=device) * noise_upper_bound

                    fake_imgs = generator(noise, labels)
                    fake_validity = discriminator(fake_imgs.detach(), labels)
                    d_fake = fake_validity.mean()

                    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), labels)
                    d_loss = d_real + d_fake + args.lambda_gp * gradient_penalty
                    d_loss.backward()
                    optim_D.step()

                disc_losses.append(d_loss.detach().cpu().numpy())

                optim_G.zero_grad()
                if USE_PROTO:
                    mu_z, sigma = generator.proto(labels)
                    eps = torch.randn_like(mu_z)
                    noise = mu_z + sigma * eps
                else:
                    noise = torch.randn(batch_size, nz, device=device) * noise_upper_bound

                fake_imgs = generator(noise, labels)
                fake_validity = discriminator(fake_imgs, labels)
                g_loss = -fake_validity.mean()

                if USE_PROTO and args.lambda_proto_reg > 0:
                    reg_term = prototype_regularizer(generator,
                        lambda_proto_reg=args.lambda_proto_reg,
                        margin=float(args.proto_reg_margin),
                        normalize=bool(args.proto_reg_normalize)
                    )
                    if reg_term is not None:
                        g_loss = g_loss + reg_term


                g_loss.backward()
                optim_G.step()
                gen_losses.append(g_loss.detach().cpu().numpy())

            else:
                optim_D.zero_grad()
                real_preds = discriminator(real_imgs, labels).view(-1)
                d_real_loss = criterion(real_preds, torch.ones_like(real_preds))

                if USE_PROTO:
                    mu_z, sigma = generator.proto(labels)
                    eps = torch.randn_like(mu_z)
                    noise = mu_z + sigma * eps
                else:
                    noise = torch.randn(batch_size, nz, device=device) * noise_upper_bound

                fake_imgs = generator(noise, labels)
                fake_preds = discriminator(fake_imgs.detach(), labels).view(-1)
                d_fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
                d_loss = d_real_loss + d_fake_loss
                disc_losses.append(d_loss.detach().cpu().numpy())
                d_loss.backward(); optim_D.step()

                optim_G.zero_grad()
                if USE_PROTO:
                    mu_z, sigma = generator.proto(labels)
                    eps = torch.randn_like(mu_z)
                    noise = mu_z + sigma * eps
                else:
                    noise = torch.randn(batch_size, nz, device=device) * noise_upper_bound
                fake_imgs = generator(noise, labels)
                fake_preds = discriminator(fake_imgs, labels).view(-1)
                g_loss = criterion(fake_preds, torch.ones_like(fake_preds))

                if USE_PROTO and args.lambda_proto_reg > 0:
                    reg_term = prototype_regularizer(generator,
                        lambda_proto_reg=args.lambda_proto_reg,
                        margin=float(args.proto_reg_margin),
                        normalize=bool(args.proto_reg_normalize)
                    )
                    if reg_term is not None:
                        g_loss = g_loss + reg_term

                gen_losses.append(g_loss.detach().cpu().numpy())
                g_loss.backward(); optim_G.step()

            if original_ratio is None:
                original_ratio = d_loss.detach().cpu().numpy() / max(1e-8, g_loss.detach().cpu().numpy())
            noise_upper_bound = get_noise_upper_bound(g_loss, d_loss, original_ratio)
            upper_bounds.append(noise_upper_bound)

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{start_epoch + num_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} noise_ub: {noise_upper_bound:.4f}")

            if i % 250 == 0:
                with torch.no_grad():
                    sample_labels = torch.arange(num_classes, device=device)
                    sample_labels = sample_labels.repeat((batch_size // num_classes) + 1)[:batch_size]
                    if USE_PROTO:
                        mu_z, sigma = generator.proto(sample_labels)
                        eps = torch.randn_like(mu_z)
                        noise = mu_z + sigma * eps
                    else:
                        noise = torch.randn(batch_size, nz, device=device)
                    fake_imgs = generator(noise, sample_labels).detach().to('cpu').numpy()
                    fake_imgs = scale_data(fake_imgs, [0, 1]).reshape(-1, 3, input_size, input_size)
                    save_dir = os.path.join(args.output_dir, args.dataset, f'v{args.version}', 'generated_img')
                    os.makedirs(save_dir, exist_ok=True)
                    grid = vutils.make_grid(torch.tensor(fake_imgs), nrow=int(math.sqrt(batch_size)))
                    plt.imsave(os.path.join(save_dir, f"epoch_{epoch}_step_{i}.png"),
                               grid.permute(1,2,0).numpy())

        if (epoch + 1) % 10 == 0:
            ckpt.save_checkpoint(generator, discriminator, optim_G, optim_D, epoch, g_loss.item(), d_loss.item())

# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def generate_images(n_samples, version, class_labels=None, offset=0):
    generator, discriminator, optim_G, optim_D = _build_models_for_infer()
    ckpt = CheckpointManager(args.dataset, version)
    ckpt.load_checkpoint(generator, discriminator, optim_G, optim_D)  # latest by default
    generator.eval()

    # build full label list of length n_samples
    if class_labels is None:
        labels = torch.arange(num_classes).repeat((n_samples // num_classes) + 1)[:n_samples]
    else:
        base = torch.tensor(class_labels, dtype=torch.long)
        if base.numel() == 0:
            raise ValueError("class_labels is empty")
        reps = (n_samples + base.numel() - 1) // base.numel()
        labels = base.repeat(reps)[:n_samples]
    labels = labels.to(device)

    save_dir = os.path.join(args.output_dir, args.dataset, f'v{version}', 'generated_img_per_class')
    os.makedirs(save_dir, exist_ok=True)

    idx_global = 0
    step = args.batch_size
    for i in range(0, n_samples, step):
        current_labels = labels[i:i+step]
        if current_labels.numel() == 0:
            break
        B = current_labels.size(0)

        if args.use_proto:
            mu_z, sigma = generator.proto(current_labels)
            eps = torch.randn_like(mu_z)
            noise = mu_z + sigma * eps
        else:
            noise = torch.randn(B, nz, device=device)

        fake = generator(noise, current_labels).detach().to('cpu').numpy()
        fake = scale_data(fake, [0, 1]).reshape(-1, 3, input_size, input_size)

        for j, img in enumerate(fake):
            lbl = int(current_labels[j].item())
            save_class_dir = os.path.join(save_dir, str(lbl))
            os.makedirs(save_class_dir, exist_ok=True)
            plt.imsave(
                os.path.join(save_class_dir, f"image_{idx_global+offset:05d}.png"),
                img.transpose(1, 2, 0)
            )
            idx_global += 1

def labels_n_per_class(n_per_class: int) -> torch.Tensor:
    return torch.arange(num_classes).repeat_interleave(n_per_class)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if args.mode == 'train':
        n_critic = 3 if args.gan_type == 'wgan-gp' else 1
        train_qgan(num_epochs=200, n_critic=n_critic)
    elif args.mode == 'generate':
        labels = labels_n_per_class(500)
        generate_images(
            n_samples=labels.numel(),
            version=args.version,
            class_labels=labels.tolist(),
            offset=0
        )

