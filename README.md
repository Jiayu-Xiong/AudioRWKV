# AudioRWKV: Efficient and Stable Bidirectional RWKV for Audio Pattern Recognition

This repository publishes **AudioSet-2M** pretrained checkpoints for three scales only: **A-RWKV-T**, **A-RWKV-S**, and **A-RWKV-B**.
 A-RWKV combines linear-time global sequence modeling (Bi-WKV) with 2D locality via depthwise separable ConvShift, making it compute-efficient and spectrogram-friendly.

------

## 🔗 Model Zoo (AudioSet-2M Pretraining)

| Variant      | Training/Validation Log(Tensorboard)                                      | Model Weights (`.pth`)                                       | Optimizer State                                              |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **A-RWKV-S** | [Log](https://drive.google.com/file/d/1Czz8TUNC2zjQJ0Hm6xxtCQyN7EzIW3pq/view?usp=drive_link) | [Model `.pth`](https://drive.google.com/file/d/1yU3umH5BwVCpJA5sZgF53R1Gg1IYOW1Q/view?usp=drive_link) | [State](https://drive.google.com/file/d/1nTubGtVptCqFHXXYBmuTgotD7xR6j6n4/view?usp=drive_link) |
| **A-RWKV-B** | [Log](https://drive.google.com/file/d/1cNADVp2U_LZlRFVdWiaaHVzBtdtH8DUs/view?usp=drive_link) | [Model `.pth`](https://drive.google.com/file/d/1TOL-j5iU1U2BPRgEV6q2Ro1Gql6Bvl47/view?usp=drive_link) | [State](https://drive.google.com/file/d/1S0hpEwMqPhY9Vj4RGNra8KfqwFJF5W0O/view?usp=drive_link) |

We do not provide the fine-tuning results of other small datasets for the time being. For datasets such as VGGSound that may have different acquisition times, we will publish the cleaned csv files and dataloader. As far as we know, it is very difficult to obtain samples of YouTube through yt-dlp again. To respect user privacy, we are also unable to provide samples.


------

## 🗂 Data Indices 

| Purpose                                        | File name                           | Google Drive link                              |
| ---------------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| Training index                                 | `un_train_index_cleaned.csv`        | [Download](https://drive.google.com/file/d/1eyzS18WZReNJ0OupQw473f6nUAaCiI0X/view?usp=drive_link) |
| Validation index                               | `eval_index_cleaned.csv`            | [Download](https://drive.google.com/file/d/1hsn7rvno5SQHVw-vlw35KWjCONMFJZlY/view?usp=drive_link) |
| Class labels                                   | `class_labels_indices.csv`          | [Download](https://drive.google.com/file/d/1b429XtsT1VPkagpVuBY51pVrYo5gUlli/view?usp=drive_link) |
| (Optional) train weights for balanced sampling | `un_train_index_cleaned_weight.csv` | [Download](https://drive.google.com/file/d/1nYyZf9V7rQiGt6grL2O6p09lh-OMH1rR/view?usp=drive_link) |

**Data roots (local paths):** `unbal/` (train), `eval/` (val)

“train weights for balanced sampling” can also be generated through utilities/get_weight.py.

------

## 🛠 Environment

- Python 3.9+; PyTorch (bfloat16 support recommended).
- Typical deps: `torch`, `timm`, `numpy`.
- Entrypoint: `run.py` for training and evaluation.

```bash
pip install torch timm numpy
```

------

## 🚀 Quick Start (fine-tuning / evaluation)

```bash
python run.py \
  --exp-dir outputs/audiorwkv_B16_audioset2m \
  --root-train unbal/ --root-val eval/ \
  --data-train un_train_index_cleaned.csv \
  --data-val   eval_index_cleaned.csv \
  --label-csv  class_labels_indices.csv \
  --batch_size 32 --accum_iter 32 \
  --epochs 25 --blr 2e-5 --warmup_epochs 10
```

------

## 📥 Loading Checkpoints (matches this repository’s save format)

During training, this codebase **saves raw `state_dict` files** (no wrapper dict) to:

- model: `<exp_dir>/models/audio_model.pth`
- optimizer (on large datasets such as AudioSet-2M): `<exp_dir>/models/optim_state.pth`
  These conventions come directly from the training loop in `traintest.py`. 

### A) Load model weights for inference

```python
import torch
from models.rwkv7 import RWKV  # adjust import to your repo

# 1) Construct the model with the exact hyperparameters used for the checkpoint.
args = ...  # must define n_layer, n_embd, ctx_len, num_classes, etc.
model = RWKV(args)
model = model.bfloat16() if torch.cuda.is_available() else model.float()

# 2) Load a raw state_dict (best or a specific epoch).
ckpt_path = "your_model.pth"  # or audio_model.{E}.pth
state_dict = torch.load(ckpt_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)  # strict=True if you prefer exact match
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()
```

### B) Resume training with optimizer state (if available)

```python
import torch

# model as above, then:
model_ckpt = "model.pth"
optim_ckpt = "optim.pth"

model.load_state_dict(torch.load(model_ckpt, map_location="cpu"), strict=True)

# Recreate the optimizer exactly as in training:
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5, weight_decay=1e-4, betas=(0.9, 0.95)
)

# Load the raw optimizer state_dict:
optimizer.load_state_dict(torch.load(optim_ckpt, map_location="cpu"))
```

> Notes
> • Checkpoint files here are **pure `state_dict`s** rather than wrapped dicts (e.g., no `{'model': …, 'optimizer': …}`), so you **load them directly** into `model.load_state_dict(…)` or `optimizer.load_state_dict(…)`. 
> • For evaluation-only usage, optimizer states are unnecessary.
> • If you changed module names or added/remapped keys, use `strict=False` and/or adapt key prefixes before loading.

------

## 📁 Repo Structure (essentials)

- `run.py` — training/validation entry with argument parsing, data pipeline, logging, and checkpointing.
- `models/rwkv7.py` — A-RWKV implementation. We have provided a simple implementation of gated bidirectional scanning in this file.
- `dataloader/` — AudioSet dataset & spectrogram utilities.
- `traintest.py` — training loop and saving policy (see Load/Resume section). 
- `cuda/wkv7.cu`— the cuda kernel of RWKV7, exactly the same as the official one, and out impl. operator will be released later. 

------

## 🖊️ Citation

If you find this work useful, please cite:

```bibtex
@article{audiorwkv2025,
  title   = {AudioRWKV: Efficient and Stable Bidirectional RWKV for Audio Pattern Recognition},
  author  = {Xiong, Jiayu and Xue, Jun and Kwan, Jianlong and Wang, Jing},
  journal = {arXiv preprint arXiv:XXXXX},
  year    = {2025}
}
```

------

## 📜 License

Please add your preferred license (e.g., Apache-2.0 or MIT) at the repository root.

