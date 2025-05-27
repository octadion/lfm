#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import urllib.request
import zipfile
import shutil
from typing import Dict, List, Tuple, Optional

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_status(status: str, message: str, details: str = ""):
    symbols = {
        "success": f"{Colors.GREEN}✓{Colors.RESET}",
        "error": f"{Colors.RED}✗{Colors.RESET}",
        "warning": f"{Colors.YELLOW}⚠{Colors.RESET}",
        "info": f"{Colors.BLUE}ℹ{Colors.RESET}"
    }
    print(f"{symbols.get(status, '')} {message}")
    if details:
        print(f"  {Colors.YELLOW}{details}{Colors.RESET}")

def check_python_version() -> Tuple[bool, str]:
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        return True, version_str
    return False, version_str

def check_cuda_availability() -> Dict[str, any]:
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            return {
                "available": True,
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        else:
            return {"available": False}
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}

def check_package_installed(package_name: str) -> Tuple[bool, Optional[str]]:
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def check_memory() -> Dict[str, float]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "percent_used": mem.percent
        }
    except ImportError:
        return {"error": "psutil not installed"}

def create_directory_structure():
    directories = [
        "checkpoints",
        "logs",
        "data",
        "outputs",
        "visualizations"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    return directories

def download_minimal_test_data():
    data_path = Path("data/test_data.txt")
    
    if data_path.exists():
        return True, "Test data already exists"
    
    test_sentences = [
        "The fundamental principles of artificial intelligence are complex.",
        "Machine learning models require significant computational resources.",
        "Natural language processing has advanced significantly in recent years.",
        "Deep learning architectures continue to evolve rapidly.",
        "Transformer models have revolutionized the field of NLP."
    ] * 20
    
    data_path.parent.mkdir(exist_ok=True)
    with open(data_path, 'w') as f:
        f.write('\n'.join(test_sentences))
    
    return True, f"Created test data with {len(test_sentences)} sentences"

def create_minimal_config():
    config = {
        "model": {
            "vocab_size": 1000,
            "embed_size": 128,
            "num_heads": 4,
            "num_experts": 2,
            "num_layers": 2,
            "max_seq_len": 128,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "max_steps": 1000,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "save_steps": 100,
            "eval_steps": 50
        },
        "data": {
            "train_file": "data/test_data.txt",
            "val_split": 0.1,
            "max_length": 128
        }
    }
    
    with open("config_minimal.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def install_requirements(requirements: List[str], use_conda: bool = False) -> List[Tuple[str, bool, str]]:
    results = []
    
    for requirement in requirements:
        try:
            if use_conda:
                cmd = ["conda", "install", "-y", requirement]
            else:
                cmd = [sys.executable, "-m", "pip", "install", requirement]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                results.append((requirement, True, "Installed successfully"))
            else:
                results.append((requirement, False, result.stderr))
        except Exception as e:
            results.append((requirement, False, str(e)))
    
    return results

def test_minimal_setup():
    test_code = """
import torch
from liquid_transformers_lm import LiquidTransformerLM

# Minimal config
model = LiquidTransformerLM(
    vocab_size=100,
    embed_size=32,
    num_heads=2,
    num_experts=2,
    num_layers=1,
    max_seq_len=64
)

# Test forward pass
input_ids = torch.randint(0, 100, (2, 16))
logits, states, aux = model(input_ids)
print(f"Model output shape: {logits.shape}")
print("Minimal setup test passed!")
"""
    
    try:
        exec(test_code)
        return True, "Minimal model test passed"
    except Exception as e:
        return False, str(e)

def run_diagnostics():
    print_header("LIQUID TRANSFORMER SETUP DIAGNOSTICS")
    
    print(f"{Colors.BOLD}System Information:{Colors.RESET}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Architecture: {platform.machine()}")
    
    print(f"\n{Colors.BOLD}1. Python Version Check:{Colors.RESET}")
    py_ok, py_version = check_python_version()
    if py_ok:
        print_status("success", f"Python {py_version} (>=3.8 required)")
    else:
        print_status("error", f"Python {py_version} (3.8+ required)")
    
    # CUDA/GPU
    print(f"\n{Colors.BOLD}2. GPU/CUDA Check:{Colors.RESET}")
    cuda_info = check_cuda_availability()
    if cuda_info.get("available"):
        print_status("success", f"CUDA {cuda_info['version']} available")
        print(f"  Device: {cuda_info['device_name']}")
        print(f"  Memory: {cuda_info['memory_gb']:.1f} GB")
    else:
        print_status("warning", "No GPU detected - CPU training will be slow")
    
    # Memory
    print(f"\n{Colors.BOLD}3. System Memory:{Colors.RESET}")
    mem_info = check_memory()
    if "total_gb" in mem_info:
        status = "success" if mem_info["total_gb"] >= 16 else "warning"
        print_status(status, f"RAM: {mem_info['total_gb']:.1f} GB total, {mem_info['available_gb']:.1f} GB free")
    
    # Required packages
    print(f"\n{Colors.BOLD}4. Required Packages:{Colors.RESET}")
    packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.30.0"),
        ("datasets", "2.10.0"),
        ("numpy", "1.24.0"),
        ("tqdm", "4.65.0"),
        ("loguru", "0.7.0"),
        ("wandb", "0.15.0"),
        ("matplotlib", "3.7.0")
    ]
    
    missing_packages = []
    for package, min_version in packages:
        installed, version = check_package_installed(package)
        if installed:
            print_status("success", f"{package} {version}")
        else:
            print_status("error", f"{package} (not installed)")
            missing_packages.append(package)
    
    # Directory structure
    print(f"\n{Colors.BOLD}5. Directory Structure:{Colors.RESET}")
    dirs = create_directory_structure()
    print_status("success", f"Created {len(dirs)} directories")
    
    # Test data
    print(f"\n{Colors.BOLD}6. Test Data:{Colors.RESET}")
    data_ok, data_msg = download_minimal_test_data()
    status = "success" if data_ok else "error"
    print_status(status, data_msg)
    
    # Create config
    print(f"\n{Colors.BOLD}7. Configuration:{Colors.RESET}")
    config = create_minimal_config()
    print_status("success", "Created minimal configuration file")
    
    # Test minimal setup
    print(f"\n{Colors.BOLD}8. Model Import Test:{Colors.RESET}")
    setup_ok, setup_msg = test_minimal_setup()
    status = "success" if setup_ok else "error"
    print_status(status, setup_msg)
    
    # Summary
    print_header("SETUP SUMMARY")
    
    all_good = py_ok and len(missing_packages) == 0 and setup_ok
    
    if all_good:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Environment is ready for training!{Colors.RESET}")
        print("\nNext steps:")
        print("1. Run comprehensive tests: python comprehensive_testing_suite.py")
        print("2. Start minimal training: python complete_training_script.py")
        print("3. Monitor with: tail -f logs/training.log")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some issues need to be fixed:{Colors.RESET}")
        
        if not py_ok:
            print(f"\n{Colors.YELLOW}• Install Python 3.8 or higher{Colors.RESET}")
        
        if missing_packages:
            print(f"\n{Colors.YELLOW}• Install missing packages:{Colors.RESET}")
            print(f"  pip install {' '.join(missing_packages)}")
        
        if not setup_ok:
            print(f"\n{Colors.YELLOW}• Check model files are present and correct{Colors.RESET}")
            print(f"  - liquid_transformers_lm.py")
            print(f"  - training_lm.py")
    
    return all_good

def interactive_setup():
    print_header("LIQUID TRANSFORMER INTERACTIVE SETUP")
    
    print("This assistant will help you set up the Liquid Transformer.\n")
    
    response = input("Would you like to automatically install missing packages? (y/n): ").lower()
    auto_install = response == 'y'
    
    print("\nRunning diagnostics...")
    all_good = run_diagnostics()
    
    if not all_good and auto_install:
        print(f"\n{Colors.YELLOW}Attempting to fix issues...{Colors.RESET}")
        
        packages = ["torch", "transformers", "datasets", "numpy", "tqdm", "loguru", "wandb", "matplotlib"]
        missing = [pkg for pkg in packages if not check_package_installed(pkg)[0]]
        
        if missing:
            print(f"\nInstalling {len(missing)} missing packages...")
            results = install_requirements(missing)
            
            for pkg, success, msg in results:
                if success:
                    print_status("success", f"Installed {pkg}")
                else:
                    print_status("error", f"Failed to install {pkg}: {msg}")
        
        print("\nRe-running diagnostics...")
        all_good = run_diagnostics()

    if all_good:
        print(f"\n{Colors.GREEN}Setup complete! You're ready to start training.{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}Please fix the remaining issues manually.{Colors.RESET}")

def create_test_script():
    test_script = '''#!/usr/bin/env python3
"""
Quick test script for Liquid Transformer
"""

import torch
from liquid_transformers_lm import LiquidTransformerLM
from transformers import AutoTokenizer

print("Loading model...")
model = CompleteLiquidTransformerLM(
    vocab_size=1000,
    embed_size=128,
    num_heads=4,
    num_experts=2,
    num_layers=2
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Testing generation...")
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate
model.eval()
with torch.no_grad():
    generated = model.generate(input_ids, max_length=50, temperature=0.8)

# Decode
output = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"\\nPrompt: {prompt}")
print(f"Generated: {output}")
print("\\nTest completed successfully!")
'''
    
    with open("test_model.py", 'w') as f:
        f.write(test_script)
    
    os.chmod("test_model.py", 0o755) 
    print_status("success", "Created test_model.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Liquid Transformer Setup Assistant")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive setup")
    parser.add_argument("--install", action="store_true", help="Auto-install missing packages")
    parser.add_argument("--create-scripts", action="store_true", help="Create helper scripts")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_setup()
    elif args.create_scripts:
        create_test_script()
        create_minimal_config()
        create_directory_structure()
        print_status("success", "Created all helper files")
    else:
        success = run_diagnostics()
        
        if args.install and not success:
            print("\nUse --interactive flag for automatic installation")
        
        sys.exit(0 if success else 1)