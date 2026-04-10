"""
Environment sanity check — runs on login node (CPU only).
Tests all imports, model instantiation, and dataset access.
Run from the project root: python test_env.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

PASS = "[OK]"
FAIL = "[FAIL]"

def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}")
        print(f"         {e}")
        return False

print("\n=== 1. Core imports ===")
check("torch",          lambda: __import__("torch"))
check("torchvision",    lambda: __import__("torchvision"))
check("numpy",          lambda: __import__("numpy"))
check("tqdm",           lambda: __import__("tqdm"))
check("einops",         lambda: __import__("einops"))
check("matplotlib",     lambda: __import__("matplotlib"))
check("PIL",            lambda: __import__("PIL"))
check("transformers",   lambda: __import__("transformers"))
check("packaging",      lambda: __import__("packaging"))
check("ninja",          lambda: __import__("ninja"))
check("triton",         lambda: __import__("triton"))

print("\n=== 2. Mamba imports ===")
check("mamba_ssm",      lambda: __import__("mamba_ssm"))
check("causal_conv1d",  lambda: __import__("causal_conv1d"))

print("\n=== 3. Project imports ===")
check("utils.vit_utils (ViTStatesExtractor)",
      lambda: __import__("utils.vit_utils", fromlist=["ViTStatesExtractor"]))

check("MambaFormer.MambaFormer (MambaFormer_Large)",
      lambda: __import__("MambaFormer.MambaFormer", fromlist=["MambaFormer_Large"]))

check("MambaFormer.utils (compute_ssd_attention_map)",
      lambda: __import__("MambaFormer.utils", fromlist=["compute_ssd_attention_map"]))

print("\n=== 4. Model instantiation (CPU) ===")
def load_teacher():
    from torchvision.models import vit_l_16
    m = vit_l_16(weights=None)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"         ViT-L/16: {params:.1f}M params")

def load_student():
    from MambaFormer.MambaFormer import MambaFormer_Large_expand1_Mamba2
    m = MambaFormer_Large_expand1_Mamba2(double_cls_token=True)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"         MambaFormer_Large_expand1_Mamba2: {params:.1f}M params")

check("ViT-L/16 teacher",                          load_teacher)
check("MambaFormer_Large_expand1_Mamba2 student",  load_student)

print("\n=== 5. Teacher weights ===")
def check_teacher_weights():
    path = os.path.join(ROOT, "models/vit/vit_l_16-852ce7e3.pth")
    assert os.path.exists(path), f"Not found: {path}"
    size = os.path.getsize(path) / 1e9
    print(f"         Found ({size:.2f} GB)")

check("vit_l_16-852ce7e3.pth", check_teacher_weights)

print("\n=== 6. Dataset access ===")
def check_dataset():
    train = os.path.join(ROOT, "dataset/ImageNet_ILSVRC2012/train")
    val   = os.path.join(ROOT, "dataset/ImageNet_ILSVRC2012/val")
    assert os.path.isdir(train), f"Not found: {train}"
    assert os.path.isdir(val),   f"Not found: {val}"
    n_train = len(os.listdir(train))
    n_val   = len(os.listdir(val))
    print(f"         train: {n_train} classes | val: {n_val} classes")
    assert n_train == 1000, f"Expected 1000 train classes, got {n_train}"
    assert n_val   == 1000, f"Expected 1000 val classes, got {n_val}"

check("ImageNet symlink + 1000 classes", check_dataset)

print("\n=== 7. Torch info ===")
import torch
print(f"  torch:        {torch.__version__}")
print(f"  CUDA build:   {torch.version.cuda}")
print(f"  CUDA avail:   {torch.cuda.is_available()} (expected False on login node)")
if torch.cuda.is_available():
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")

print()
