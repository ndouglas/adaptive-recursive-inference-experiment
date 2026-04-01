import torch
import transformers

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available() if torch.backends.mps.is_available() else "N/A")
    print("MPS built:", torch.backends.mps.is_built() if torch.backends.mps.is_built() else "N/A")
    print("Transformers version:", transformers.__version__)

    a = torch.randn(3, 3)
    b = torch.randn(3, 3)

    expected = a @ b

    if torch.backends.mps.is_available():
      result = (a.to("mps") @ b.to("mps")).to("cpu")
      assert torch.allclose(result, expected, atol=1e-6), "MPS matrix multiplication result does not match expected value."
      print("MPS matrix multiplication test passed.")
    else:
      print("MPS not available, skipping MPS matrix multiplication test.")


if __name__ == "__main__":
    main()