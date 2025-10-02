#!/usr/bin/env python3
"""
Download real ONNX models for integration testing.

This script downloads pre-trained ONNX models from the ONNX Model Zoo
and HuggingFace for comprehensive integration testing of RONN.

Models:
- ResNet18: Computer vision (image classification)
- DistilBERT: NLP (text embeddings/classification)
- GPT-2 Small: Text generation (autoregressive)
"""

import sys
import os
import urllib.request
from pathlib import Path


MODELS = {
    'resnet18': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx',
        'filename': 'resnet18.onnx',
        'size_mb': 45,
        'description': 'ResNet-18 image classification (ImageNet)',
    },
    # Note: For BERT and GPT-2, we'll need to convert from HuggingFace or use smaller variants
    # These URLs are placeholders - actual models may need to be exported using transformers library
}


def download_file(url, filename, expected_size_mb=None):
    """Download a file with progress indicator."""
    print(f"Downloading {filename} from {url}...")

    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        urllib.request.urlretrieve(url, filename, reporthook=reporthook)
        print()  # New line after progress

        # Check file size
        actual_size = os.path.getsize(filename) / 1024 / 1024
        print(f"  ✓ Downloaded {filename} ({actual_size:.1f} MB)")

        return True
    except Exception as e:
        print(f"\n  ✗ Failed to download {filename}: {e}")
        return False


def check_existing_model(filename, expected_size_mb=None):
    """Check if model already exists and is valid."""
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"  ℹ Model already exists: {filename} ({size_mb:.1f} MB)")

        # Basic validation - check if file is not empty and roughly expected size
        if size_mb > 0:
            return True
    return False


def download_resnet18():
    """Download ResNet-18 model from ONNX Model Zoo."""
    info = MODELS['resnet18']
    filename = info['filename']

    print(f"\n1. ResNet-18 ({info['description']})")

    if check_existing_model(filename, info['size_mb']):
        print("  → Skipping download (already exists)")
        return True

    return download_file(info['url'], filename, info['size_mb'])


def create_distilbert_script():
    """Create a helper script to export DistilBERT from HuggingFace."""
    script_content = '''#!/usr/bin/env python3
"""
Export DistilBERT to ONNX format.

Requires: pip install transformers torch onnx
"""

import torch
from transformers import DistilBertModel, DistilBertTokenizer

def export_distilbert():
    model_name = "distilbert-base-uncased"

    print(f"Loading {model_name}...")
    model = DistilBertModel.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    model.eval()

    # Create dummy input
    dummy_input = tokenizer("This is a sample sentence for export", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "distilbert.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=13,
    )

    print("✓ Exported distilbert.onnx")

    # Print model info
    import os
    size_mb = os.path.getsize("distilbert.onnx") / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    export_distilbert()
'''

    script_path = "export_distilbert.py"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"\n2. DistilBERT (NLP embeddings)")
    print(f"  ℹ Created {script_path}")
    print("  → To export: python export_distilbert.py")
    print("     Requires: pip install transformers torch onnx")


def create_gpt2_script():
    """Create a helper script to export GPT-2 Small from HuggingFace."""
    script_content = '''#!/usr/bin/env python3
"""
Export GPT-2 Small to ONNX format.

Requires: pip install transformers torch onnx
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer

def export_gpt2():
    model_name = "gpt2"  # This is GPT-2 Small (124M params)

    print(f"Loading {model_name}...")
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.eval()

    # Create dummy input
    dummy_text = "This is a sample text for GPT-2 export"
    dummy_input = tokenizer(dummy_text, return_tensors="pt")
    input_ids = dummy_input["input_ids"]

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        input_ids,
        "gpt2-small.onnx",
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=13,
    )

    print("✓ Exported gpt2-small.onnx")

    # Print model info
    import os
    size_mb = os.path.getsize("gpt2-small.onnx") / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    export_gpt2()
'''

    script_path = "export_gpt2.py"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"\n3. GPT-2 Small (text generation)")
    print(f"  ℹ Created {script_path}")
    print("  → To export: python export_gpt2.py")
    print("     Requires: pip install transformers torch onnx")


def main():
    print("=" * 60)
    print("RONN Integration Test Model Downloader")
    print("=" * 60)

    # Ensure we're in the models directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"\nWorking directory: {os.getcwd()}")

    # Download ResNet-18 (direct download available)
    resnet_ok = download_resnet18()

    # Create export scripts for transformer models
    # (These require local PyTorch/Transformers installation)
    create_distilbert_script()
    create_gpt2_script()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if resnet_ok:
        print("✓ ResNet-18: Ready for testing")
    else:
        print("✗ ResNet-18: Download failed")

    print("ℹ DistilBERT: Run export_distilbert.py to create model")
    print("ℹ GPT-2 Small: Run export_gpt2.py to create model")

    print("\nNext steps:")
    print("1. Install dependencies (if needed):")
    print("   pip install transformers torch onnx")
    print("2. Export transformer models:")
    print("   python export_distilbert.py")
    print("   python export_gpt2.py")
    print("3. Run integration tests:")
    print("   cargo test --test integration -- --include-ignored")

    return 0 if resnet_ok else 1


if __name__ == '__main__':
    sys.exit(main())
