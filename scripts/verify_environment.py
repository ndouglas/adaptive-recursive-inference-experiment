import sys
import torch
import transformers


def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA: not available")
        return False

    print(f"CUDA: available ({torch.version.cuda})")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        vram_gb = total_mem / (1024 ** 3)
        print(f"  GPU {i}: {props.name} — {vram_gb:.1f} GB VRAM, compute {props.major}.{props.minor}")

    # Matmul smoke test
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    assert c.shape == (256, 256), "CUDA matmul failed"
    print("  CUDA matmul test: passed")
    return True


def check_mps():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("MPS: not available")
        return False

    print("MPS: available")
    a = torch.randn(256, 256, device="mps")
    b = torch.randn(256, 256, device="mps")
    c = (a @ b).to("cpu")
    assert c.shape == (256, 256), "MPS matmul failed"
    print("  MPS matmul test: passed")
    return True


def check_model_loading(device):
    """Quick smoke test: load a tiny model config and verify it lands on the right device."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")
    print(f"  Model config loaded: {config.num_hidden_layers} layers, "
          f"{config.hidden_size} hidden, {config.vocab_size} vocab")
    return True


def estimate_model_memory(num_params_b, dtype_bytes=2):
    """Estimate VRAM needed for a model in GB (weights + ~20% overhead for KV cache/activations)."""
    weight_gb = num_params_b * dtype_bytes
    return weight_gb * 1.2


def main():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print()

    has_cuda = check_cuda()
    has_mps = check_mps()

    if not has_cuda and not has_mps:
        print("\nNo GPU accelerator available — CPU only.")

    print()
    check_model_loading("cuda" if has_cuda else "mps" if has_mps else "cpu")

    # VRAM budget summary for RunPod planning
    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        vram_gb = (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / (1024 ** 3)
        print(f"\nVRAM budget ({vram_gb:.0f} GB):")
        for name, params in [("Qwen2.5-1.5B", 1.5), ("Qwen2.5-7B", 7.6), ("Qwen2.5-14B", 14.8)]:
            est = estimate_model_memory(params)
            fits = "fits" if est < vram_gb else "DOES NOT FIT"
            print(f"  {name}: ~{est:.1f} GB fp16 — {fits}")

    print("\nEnvironment OK.")


if __name__ == "__main__":
    main()
