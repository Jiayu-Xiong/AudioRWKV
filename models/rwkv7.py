########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

'''
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

args = types.SimpleNamespace()

# DTYPE = torch.bfloat16
DTYPE = torch.half # better

args.head_size_a = 64 # don't change
HEAD_SIZE = args.head_size_a

USE_CUDA_KERNEL = True # False => UNOPTIMIZED, VERY SLOW

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

# flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
# load(name="wind_backstepping_bi", sources=[f'cuda/wkv7_cuda_fb.cu', 'cuda/wkv7_op_bi.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b,y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db
    

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

# disabled
class WindBackstepping_BiT(torch.autograd.Function):
    """
    PyTorch Autograd Function for the bidirectional Wind Backstepping (RWKV-like) model.
    Handles both forward and backward passes by calling custom CUDA kernels.
    """

    @staticmethod
    def forward(ctx, w, q, k, v, a, b, g):
        """
        Forward pass for the bidirectional Wind Backstepping model.

        Args:
            ctx: Context object to save tensors for backward pass.
            w (torch.Tensor): Weight tensor. Shape: (B, T, H, C)
            q (torch.Tensor): Query tensor. Shape: (B, T, H, C)
            k (torch.Tensor): Key tensor. Shape: (B, T, H, C)
            v (torch.Tensor): Value tensor. Shape: (B, T, H, C)
            a (torch.Tensor): Coefficient 'a' tensor. Shape: (B, T, H, C)
            b (torch.Tensor): Coefficient 'b' tensor. Shape: (B, T, H, C)
            g (torch.Tensor): Gate tensor (scalar per B, H). Shape: (B, H)

        Returns:
            torch.Tensor: The final merged output tensor `y`. Shape: (B, T, H, C)
        """
        # Extract dimensions from an input tensor
        B, T, H, C = w.shape

        # Assertions for input tensor properties
        assert T % CHUNK_LEN == 0, f"Sequence length T ({T}) must be a multiple of CHUNK_LEN ({CHUNK_LEN})"
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, a, b, g]), "All input tensors (w,q,k,v,a,b,g) must be of bfloat16 dtype"
        assert all(i.is_contiguous() for i in [w, q, k, v, a, b, g]), "All input tensors (w,q,k,v,a,b,g) must be contiguous"

        # Allocate output tensors and intermediate tensors for CUDA kernels
        # s_f, s_b: Chunked states. The CUDA kernel indicates they are C x C matrices per chunk.
        # This matches the (B, H, T//CHUNK_LEN, C, C) shape.
        s_f = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        s_b = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)

        # sa_f, sa_b: Scalar 'sa' values. The CUDA kernel implies these are (B, T, H, C)
        # where each C-dimension entry might store the same scalar value (as noted in prior review).
        # We follow the shape used in the C++ binding for now.
        sa_f = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        sa_b = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)

        # y: Final merged output (same shape as v)
        y = torch.empty_like(v)
        # y_f, y_b: Intermediate outputs from forward and backward scans (same shape as v)
        yf = torch.empty_like(v)
        yb = torch.empty_like(v)

        # Call the custom CUDA forward operation
        # The arguments are now correctly aligned with the C++ binding signature:
        # (w, q, k, v, a, b, y_f, y_b, y, g, s_f, s_b, sa_f, sa_b)
        torch.ops.wind_backstepping_bi.forward(w, q, k, v, a, b, yf, yb, y, g, s_f, s_b, sa_f, sa_b)

        # Save tensors needed for the backward pass.
        # `yf` and `yb` are now saved as they are explicitly passed to `backward_kernel_bwd`.
        # `sa_f` and `sa_b` are also explicitly needed by backward kernels.
        ctx.save_for_backward(w, q, k, v, a, b, g, s_f, s_b, sa_f, sa_b, yf, yb)
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for the bidirectional Wind Backstepping model.

        Args:
            ctx: Context object with saved tensors.
            dy (torch.Tensor): Gradient of the loss with respect to the output `y`.
                               Expected dtype: bfloat16, contiguous.

        Returns:
            Tuple[torch.Tensor]: Gradients for the input tensors (dw, dq, dk, dv, da, db, dg).
        """
        assert dy.dtype == torch.bfloat16, "Gradient dy must be of bfloat16 dtype"
        assert dy.is_contiguous(), "Gradient dy must be contiguous"

        # Retrieve saved tensors from the forward pass context
        # Ensure the order and names match what was saved in `save_for_backward`.
        w, q, k, v, a, b, g, s_f, s_b, sa_f, sa_b, yf, yb = ctx.saved_tensors

        # Create empty tensors for gradients to be computed by the CUDA backward operation
        dw = torch.zeros_like(w)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        da = torch.zeros_like(a)
        db = torch.zeros_like(b)
        dg = torch.zeros_like(g)

        # Call the custom CUDA backward operation
        # The arguments are now correctly aligned with the C++ binding signature:
        # (w, q, k, v, a, b, y_f, y_b, dy, s_f, s_b, sa_f, sa_b, dw, dq, dk, dv, da, db, g, dg)
        torch.ops.wind_backstepping_bi.backward(
            w, q, k, v, a, b,
            yf, yb, dy,  # y_f and y_b are now inputs to backward pass
            s_f, s_b, sa_f, sa_b, # Renamed
            dw, dq, dk, dv, da, db,
            g, # g is input
            dg # dg is output
        )
        # Return gradients for each input tensor passed to the forward method
        return dw, dq, dk, dv, da, db, dg


# Helper function to run the bidirectional layer
# disabled
def RUN_CUDA_RWKV7g_BiT(q_in, w_in, k_in, v_in, a_in, b_in, g_in):
    """
    Applies the WindBackstepping_Bi layer, handling tensor reshaping for compatibility
    with the underlying CUDA kernel's expected (B, T, H, C) dimensions.

    Args:
        q_in (torch.Tensor): Input queries. Shape: (B, T, H*C)
        w_in (torch.Tensor): Input weights. Shape: (B, T, H*C)
        k_in (torch.Tensor): Input keys. Shape: (B, T, H*C)
        v_in (torch.Tensor): Input values. Shape: (B, T, H*C)
        a_in (torch.Tensor): Input 'a' coefficients. Shape: (B, T, H*C)
        b_in (torch.Tensor): Input 'b' coefficients. Shape: (B, T, H*C)
        g_in (torch.Tensor): Input gate. Shape: (B, H)

    Returns:
        torch.Tensor: The output tensor `y`. Shape: (B, T, H*C)
    """
    B, T, HC = q_in.shape
    # Assuming C = 64 based on `HC // 64`
    H = HC // 64
    C = 64

    # Reshape input tensors from (B, T, H*C) to (B, T, H, C)
    # The gate `g_in` is already (B, H), which is its expected shape.
    q = q_in.view(B, T, H, C).contiguous()
    w = w_in.view(B, T, H, C).contiguous()
    k = k_in.view(B, T, H, C).contiguous()
    v = v_in.view(B, T, H, C).contiguous()
    a = a_in.view(B, T, H, C).contiguous()
    b = b_in.view(B, T, H, C).contiguous()

    # Ensure g_in is contiguous if it's not already, as per assertion.
    g = g_in.contiguous()

    # Apply the autograd function
    result = WindBackstepping_BiT.apply(w, q, k, v, a, b, g)

    # Reshape the output back to (B, T, H*C)
    return result.view(B, T, HC)

# disabled
class WindBackstepping_BiB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, g):
        B, T, H, C = w.shape

        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, a, b, g])
        assert all(i.is_contiguous() for i in [w, q, k, v, a, b, g])

        s_f = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        s_b = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)

        sa_f = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        sa_b = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)

        y = torch.empty_like(v)
        yf = torch.empty_like(v)
        yb = torch.empty_like(v)

        torch.ops.wind_backstepping_bi.forward(w, q, k, v, a, b, yf, yb, y, g, s_f, s_b, sa_f, sa_b)

        ctx.save_for_backward(w, q, k, v, a, b, g, s_f, s_b, sa_f, sa_b, yf, yb)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.dtype == torch.bfloat16
        assert dy.is_contiguous()

        w, q, k, v, a, b, g, s_f, s_b, sa_f, sa_b, yf, yb = ctx.saved_tensors

        dw = torch.zeros_like(w)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        da = torch.zeros_like(a)
        db = torch.zeros_like(b)
        dg = torch.zeros_like(g)

        torch.ops.wind_backstepping_bi.backward(
            w, q, k, v, a, b,
            yf, yb, dy,
            s_f, s_b, sa_f, sa_b,
            dw, dq, dk, dv, da, db,
            g,
            dg
        )
        return dw, dq, dk, dv, da, db, dg


# disabled
def RUN_CUDA_RWKV7g_BiB(q_in, w_in, k_in, v_in, a_in, b_in, g_in):
    B, T, HC = q_in.shape
    H = HC // 64
    C = 64

    q = q_in.view(B, T, H, C).contiguous()
    w = w_in.view(B, T, H, C).contiguous()
    k = k_in.view(B, T, H, C).contiguous()
    v = v_in.view(B, T, H, C).contiguous()
    a = a_in.view(B, T, H, C).contiguous()
    b = b_in.view(B, T, H, C).contiguous()
    g = g_in.contiguous()

    result = WindBackstepping_BiB.apply(w, q, k, v, a, b, g)

    return result.view(B, T, HC)


########################################################################################################
# RWKV TimeMix
########################################################################################################

@torch.compile
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        self.Hi, self.Wi = args.h, args.w

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)


            self.gate = nn.Conv1d(
                in_channels=C,
                out_channels=H,
                kernel_size=1,
                groups=H,
                bias=False
            )

            # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.time_shift = nn.Conv2d(kernel_size=(3, 3), stride=1, dilation=1, padding=(1, 1), bias=False, in_channels=C, out_channels=C, groups=C)
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!
            # self.spec_shift = nn.Conv2d(kernel_size=3, stride=1, dilation=1, padding=1, bias=False, in_channels=C, out_channels=C, groups=C)
            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.time_shift.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x.permute(0, 2, 1).view(B, C, self.Hi, self.Wi)).view(B, C, -1).permute(0, 2, 1).contiguous() - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)

        k = self.key(xk)
        v = self.value(xv)
        # v = self.spec_shift(v.permute(0, 2, 1).view(B, C, self.Hi, self.Wi))
        # v = v.view(B, C, -1).permute(0, 2, 1).contiguous()
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(torch.cat([r, torch.flip(r, dims=[1])], dim=0), torch.cat([w, torch.flip(w, dims=[1])], dim=0), # w.repeat(2, 1, 1), 
                            torch.cat([k, torch.flip(k, dims=[1])], dim=0), 
                            torch.cat([v, torch.flip(v, dims=[1])], dim=0), torch.cat([-kk, torch.flip(-kk, dims=[1])], dim=0),
                            torch.cat([(kk*a), torch.flip(kk*a, dims=[1])], dim=0)) #  -kk.repeat(2, 1, 1), (kk*a).repeat(2, 1, 1))
        x_f, x_b = torch.chunk(x.view(2 * B, T, H, -1), chunks=2, dim=0)
        gate = torch.sigmoid(self.gate(xx.transpose(1, 2))).transpose(1, 2).unsqueeze(3) # B 1 C -> B C 1 -> B H 1 -> B 1 H

        x = gate * x_f + (1.0 - gate) * torch.flip(x_b, dims=[1])

        # gate = sigmoid(self.gate)
        # x = gate*x[:B] + (1-gate)*torch.flip((x[B:]), dims=[1])
        # x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)

        # x = RUN_CUDA_RWKV7g_Bi(r, w, k, v, -kk, kk*a, self.gate(x.mean(dim=1, keepdim=True).transpose(1, 2)))
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

@torch.compile
def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0).square()

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        C = args.n_embd
        self.Hi, self.Wi = args.h, args.w
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_shift = nn.Conv2d(kernel_size=(3, 3), stride=1, dilation=1, padding=(1, 1), bias=False, in_channels=C, out_channels=C, groups=C)

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        B, L, C = x.shape
        xx = self.time_shift(x.permute(0, 2, 1).view(B, C, self.Hi, self.Wi)).view(B, C, -1).permute(0, 2, 1).contiguous() - x
        
        k = x + xx * self.x_k
        k = F.relu(self.key(k)) ** 2

        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(MyModule):
    def __init__(self, args, layer_id, dpr=None):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # self.pe = nn.Conv1d(in_channels=args.n_embd, out_channels=args.n_embd, kernel_size=13, padding=6, stride=1, groups=args.n_embd)
        # self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

        self.drop = DropPath(args.drop) if dpr == None else DropPath(dpr)
        
    @MyFunction
    def forward(self, x, v_first):
        # if self.layer_id == 0:
        #     x = self.ln0(x)
        # x = x + self.pe(x.transpose(1,2)).transpose(1,2)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + self.drop(xx)
        x = x + self.drop(self.ffn(self.ln2(x)))

        # x = self.ln1(x)
        # xx, v_first = self.att(x, v_first)
        # # x = self.ln1(x + self.drop(xx))
        # x = x + self.drop(xx)
        # x = self.ln2(x)
        # x = x + self.drop(self.ffn(x))
        # x = self.ln2(x + self.drop(self.ffn(x)))
        # xx, v_first = self.att(self.ln1(x), v_first)
        # x = x + xx
        # x = x + self.ffn(self.ln2(x))

        return x, v_first

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        # self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        assert args.num_mels % 16 ==0 and args.target_length % 16 == 0
        self.emb = nn.Conv2d(kernel_size = 16, stride = 16, in_channels = 1, out_channels = args.n_embd)
        # self.ln_in = nn.LayerNorm(args.n_embd)

        self.pos_embed = nn.Parameter(torch.zeros(1, args.num_mels//16 * args.target_length//16, args.n_embd))
        self.pos_drop = nn.Dropout(p=args.drop)
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, args.drop, args.n_layer)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([Block(args, i, dpr=dpr[i]) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        # self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.head = nn.Linear(args.n_embd, args.num_classes, bias=False)
        
        self.generate_init_weight()

    def forward(self, idx):

        x = self.emb(idx)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1).contiguous()
        # x = x.squeeze().permute(0, 2, 1)  # -> (B, H, W, C)
        # self.ln_in(x)
        x = self.pos_embed + x
        x = self.pos_drop(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        
        # x = x.reshape(x.size(0), -1, x.size(3))  # -> (B, L, C)
        x = x.contiguous()
        x = self.ln_out(x.mean(dim=1, keepdim=False))
        x = self.head(x)

        return x


    def generate_init_weight(self):
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight" or n.__contains__('time_shift') or n.__contains__('gate'):
                m[n] = p
                trunc_normal_(p, std=1e-4)
                print(f" [scale {scale}]")
            elif n.__contains__('.bias'):
                nn.init.zeros_(m[n])
            elif n == "head.weight":
                m[n] = p
                if self.args.num_classes > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.num_classes / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                if False: # if 'mamba' in os.environ["RWKV_MY_TESTING"]:
                    m[n] = p
                    if '.out_proj.weight' in n:
                        scale = 0
                        nn.init.zeros_(m[n])
                        print(f" [scale {scale}]")
                    elif '.bias' in n:
                        scale = 0
                        nn.init.zeros_(m[n])
                        print(f" [scale {scale}]")
                    else:
                        print()
                else:
                    assert n.endswith('.weight') # should always be true

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                    for kk in [".att.key."]:
                        if kk in n:
                            scale = 0.1
                    for kk in [".att.gate."]:
                        if kk in n:
                            scale = 0.1

                    print(f" [scale {scale}]")

                    if True:# self.args.accelerator.upper() == "GPU":
                        m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                    else:
                        m[n] = torch.empty((shape[0], shape[1]))

                    if scale == 0:
                        nn.init.zeros_(m[n])
                    elif scale < 0:
                        nn.init.uniform_(m[n], a=scale, b=-scale)
                    else:
                        nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            # if os.environ["RWKV_FLOAT_MODE"] == "fp16":
            #     m[n] = m[n].half()
            # elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
            #     m[n] = m[n].bfloat16()
            m[n] = m[n].bfloat16()
            n_params += m[n].numel()

            # if n == "emb.weight":
            #     print(m[n])

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
