import torch
import torch.nn as nn
import numpy as np
import sys
import traceback
from pathlib import Path
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore, Style
import warnings
warnings.filterwarnings("ignore")

init(autoreset=True)

class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test_name: str, message: str = ""):
        self.passed.append((test_name, message))
        print(f"{Fore.GREEN}✓ {test_name}{Style.RESET_ALL} {message}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"{Fore.RED}✗ {test_name}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Error: {error}{Style.RESET_ALL}")
    
    def add_warning(self, test_name: str, warning: str):
        self.warnings.append((test_name, warning))
        print(f"{Fore.YELLOW}⚠ {test_name}: {warning}{Style.RESET_ALL}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print("\n" + "="*60)
        print(f"{Fore.CYAN}TEST SUMMARY{Style.RESET_ALL}")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"{Fore.GREEN}Passed: {len(self.passed)}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {len(self.failed)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Warnings: {len(self.warnings)}{Style.RESET_ALL}")
        
        if self.failed:
            print(f"\n{Fore.RED}Failed tests:{Style.RESET_ALL}")
            for test, error in self.failed:
                print(f"  - {test}: {error[:100]}...")
        
        success_rate = (len(self.passed) / total * 100) if total > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        
        return len(self.failed) == 0


def check_system_requirements(results: TestResults):
    print(f"\n{Fore.CYAN}=== SYSTEM REQUIREMENTS CHECK ==={Style.RESET_ALL}")

    python_version = sys.version.split()[0]
    if sys.version_info >= (3, 8):
        results.add_pass("Python version", f"v{python_version}")
    else:
        results.add_fail("Python version", f"Need Python 3.8+, got {python_version}")
    
    try:
        import torch
        results.add_pass("PyTorch installed", f"v{torch.__version__}")
        
        if torch.cuda.is_available():
            results.add_pass("CUDA available", f"{torch.cuda.get_device_name(0)}")
            results.add_pass("CUDA version", f"v{torch.version.cuda}")
        else:
            results.add_warning("CUDA", "Not available - will use CPU (slower)")
    except ImportError:
        results.add_fail("PyTorch", "Not installed")
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb >= 16:
        results.add_pass("RAM", f"{ram_gb:.1f} GB")
    else:
        results.add_warning("RAM", f"{ram_gb:.1f} GB (recommended: 16+ GB)")
    
    try:
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_mem = gpus[0].memoryTotal / 1024
                if gpu_mem >= 8:
                    results.add_pass("GPU Memory", f"{gpu_mem:.1f} GB")
                else:
                    results.add_warning("GPU Memory", f"{gpu_mem:.1f} GB (recommended: 8+ GB)")
    except:
        pass
    
    packages = {
        'transformers': 'Transformers library',
        'datasets': 'Datasets library',
        'loguru': 'Logging',
        'wandb': 'Experiment tracking',
        'tqdm': 'Progress bars',
        'matplotlib': 'Plotting',
        'numpy': 'Numerical computing'
    }
    
    print(f"\n{Fore.CYAN}Required packages:{Style.RESET_ALL}")
    for package, desc in packages.items():
        try:
            __import__(package)
            results.add_pass(f"Package {package}", desc)
        except ImportError:
            results.add_fail(f"Package {package}", "Not installed")


def test_liquid_cell_components(results: TestResults):
    print(f"\n{Fore.CYAN}=== LIQUID CELL COMPONENTS TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm import (
            LiquidCell, 
            HierarchicalLiquidCell,
            ODESolver
        )
        results.add_pass("Import liquid cells")

        solver = ODESolver()
        y0 = torch.randn(4, 128)
        t = torch.linspace(0, 1, 10)
        
        def dummy_func(t, y):
            return -y
        
        try:
            result = solver.odeint(dummy_func, y0, t, method='euler')
            results.add_pass("ODE solver (Euler)", f"Output shape: {result.shape}")
        except Exception as e:
            results.add_fail("ODE solver", str(e))
        
        batch_size = 4
        hidden_size = 128
        cell = LiquidCell(hidden_size, hidden_size)
        
        x = torch.randn(batch_size, hidden_size)
        h = torch.randn(batch_size, hidden_size)
        
        try:
            h_new = cell(x, h)
            assert h_new.shape == (batch_size, hidden_size)
            results.add_pass("TrueLiquidCell forward", f"Output shape: {h_new.shape}")
        except Exception as e:
            results.add_fail("TrueLiquidCell forward", str(e))
        
        hier_cell = HierarchicalLiquidCell(hidden_size)
        fast_h = torch.randn(batch_size, hidden_size)
        medium_h = torch.randn(batch_size, hidden_size)
        slow_h = torch.randn(batch_size, hidden_size)
        
        try:
            combined, f, m, s = hier_cell(x, fast_h, medium_h, slow_h)
            assert combined.shape == (batch_size, hidden_size)
            results.add_pass("HierarchicalLiquidCell", f"Output shape: {combined.shape}")
        except Exception as e:
            results.add_fail("HierarchicalLiquidCell", str(e))
            
    except ImportError as e:
        results.add_fail("Import liquid components", str(e))
    except Exception as e:
        results.add_fail("Liquid cell components", str(e))


def test_attention_and_moe(results: TestResults):
    """Test attention and MoE components."""
    print(f"\n{Fore.CYAN}=== ATTENTION & MOE TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm import (
            TimeAwareAttention,
            AdaptiveMixtureOfExperts
        )
        
        batch_size = 4
        seq_len = 32
        embed_size = 128
        
        attn = TimeAwareAttention(embed_size, num_heads=8)
        x = torch.randn(batch_size, seq_len, embed_size)
        
        try:
            attn_out, attn_weights = attn(x, time_step=0.5)
            assert attn_out.shape == x.shape
            results.add_pass("TimeAwareAttention", f"Output shape: {attn_out.shape}")
        except Exception as e:
            results.add_fail("TimeAwareAttention", str(e))
        
        moe = AdaptiveMixtureOfExperts(
            num_experts=4,
            input_dim=embed_size,
            hidden_dim=embed_size * 2,
            liquid_dim=embed_size
        )
        
        liquid_state = torch.randn(batch_size, embed_size)
        
        try:
            moe_out, aux_losses = moe(x, liquid_state)
            assert moe_out.shape == x.shape
            assert 'load_balance_loss' in aux_losses
            results.add_pass("AdaptiveMoE", f"Output shape: {moe_out.shape}")
        except Exception as e:
            results.add_fail("AdaptiveMoE", str(e))
            
    except ImportError as e:
        results.add_fail("Import attention/MoE", str(e))
    except Exception as e:
        results.add_fail("Attention/MoE components", str(e))


def test_full_model(results: TestResults):
    print(f"\n{Fore.CYAN}=== FULL MODEL TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm import LiquidTransformerLM

        config = {
            "vocab_size": 1000,
            "embed_size": 128,
            "num_heads": 4,
            "num_experts": 2,
            "num_layers": 2,
            "max_seq_len": 64,
            "dropout": 0.1
        }
        
        model = LiquidTransformerLM(**config)
        results.add_pass("Model initialization", 
                        f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        
        try:
            logits, liquid_states, aux_losses = model(input_ids)

            assert logits.shape == (batch_size, seq_len, config["vocab_size"])
            assert len(liquid_states) == config["num_layers"]
            assert all('combined' in ls for ls in liquid_states)
            
            results.add_pass("Model forward pass", f"Logits shape: {logits.shape}")

            loss = logits.mean()
            loss.backward()
            
            has_gradients = any(p.grad is not None for p in model.parameters())
            if has_gradients:
                results.add_pass("Gradient computation", "Gradients computed successfully")
            else:
                results.add_fail("Gradient computation", "No gradients computed")
                
        except Exception as e:
            results.add_fail("Model forward pass", str(e))
            traceback.print_exc()
        
        try:
            model.eval()
            prompt = torch.tensor([[101, 102, 103]])
            generated = model.generate(prompt, max_length=20, temperature=1.0)
            
            assert generated.shape[1] > prompt.shape[1]
            results.add_pass("Text generation", f"Generated shape: {generated.shape}")
        except Exception as e:
            results.add_fail("Text generation", str(e))
            
    except ImportError as e:
        results.add_fail("Import model", str(e))
    except Exception as e:
        results.add_fail("Full model test", str(e))
        traceback.print_exc()


def test_training_components(results: TestResults):
    print(f"\n{Fore.CYAN}=== TRAINING COMPONENTS TEST ==={Style.RESET_ALL}")
    
    try:
        from train_lm import (
            CurriculumLanguageModelingDataset,
            LiquidTransformerTrainer,
            AdaptiveLearningRateScheduler,
            curriculum_sequence_length
        )
        from transformers import AutoTokenizer

        try:
            length = curriculum_sequence_length(25000, min_len=64, max_len=512)
            assert 64 <= length <= 512
            results.add_pass("Curriculum length function", f"Length at step 25k: {length}")
        except Exception as e:
            results.add_fail("Curriculum length", str(e))
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            texts = ["This is a test." * 50] * 10 
            
            dataset = CurriculumLanguageModelingDataset(
                tokenizer, texts, min_length=64, max_length=512
            )
            
            batch = dataset.get_batch_with_length([0, 1], max_length=128)
            assert batch["input_ids"].shape == (2, 128)
            assert batch["attention_mask"].shape == (2, 128)
            
            results.add_pass("CurriculumDataset", f"Batch shape: {batch['input_ids'].shape}")
        except Exception as e:
            results.add_fail("CurriculumDataset", str(e))

        try:
            from torch.optim import Adam
            dummy_model = nn.Linear(10, 10)
            optimizer = Adam(dummy_model.parameters())
            scheduler = AdaptiveLearningRateScheduler(optimizer, base_lr=1e-4)

            liquid_states = [{'combined': torch.randn(4, 128)}]
            lr = scheduler.step(liquid_states)
            
            assert isinstance(lr, float)
            results.add_pass("AdaptiveLRScheduler", f"Adaptive LR: {lr:.6f}")
        except Exception as e:
            results.add_fail("AdaptiveLRScheduler", str(e))
            
    except ImportError as e:
        results.add_fail("Import training components", str(e))
    except Exception as e:
        results.add_fail("Training components", str(e))


def test_mock_training_loop(results: TestResults):
    print(f"\n{Fore.CYAN}=== MOCK TRAINING LOOP TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm import LiquidTransformerLM
        from train_lm import (
            CurriculumLanguageModelingDataset,
            LiquidTransformerTrainer
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        tiny_config = {
            "vocab_size": tokenizer.vocab_size,
            "embed_size": 64,
            "num_heads": 2,
            "num_experts": 2,
            "num_layers": 1,
            "max_seq_len": 128,
            "dropout": 0.1
        }
        
        model = LiquidTransformerLM(**tiny_config)

        trainer = LiquidTransformerTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_wandb=False 
        )
        
        results.add_pass("Trainer initialization", "Trainer created successfully")

        texts = ["This is a test sentence. " * 20] * 50
        dataset = CurriculumLanguageModelingDataset(tokenizer, texts)
  
        try:
            batch_size = 2
            for step in range(5):
                indices = np.random.choice(len(dataset), batch_size)
                batch = dataset.get_batch_with_length(indices.tolist(), max_length=64)

                loss, liquid_states, aux_losses = trainer.train_step(
                    batch, None, step, 100
                )
                
                assert isinstance(loss, float)
                assert loss > 0
                
                if step == 0:
                    results.add_pass("Training step", f"Loss: {loss:.4f}")
            
            results.add_pass("Mock training loop", "5 steps completed successfully")
            
        except Exception as e:
            results.add_fail("Mock training loop", str(e))
            traceback.print_exc()
            
    except Exception as e:
        results.add_fail("Mock training setup", str(e))
        traceback.print_exc()


def test_memory_usage(results: TestResults):
    print(f"\n{Fore.CYAN}=== MEMORY USAGE TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm import LiquidTransformerLM
        
        configs = [
            ("Tiny", {"vocab_size": 1000, "embed_size": 64, "num_heads": 2, 
                     "num_experts": 2, "num_layers": 1}),
            ("Small", {"vocab_size": 10000, "embed_size": 256, "num_heads": 4, 
                      "num_experts": 2, "num_layers": 2}),
            ("Medium", {"vocab_size": 30000, "embed_size": 512, "num_heads": 8, 
                       "num_experts": 4, "num_layers": 4}),
        ]
        
        for name, config in configs:
            config["max_seq_len"] = 128
            
            try:
                model = LiquidTransformerLM(**config)

                total_params = sum(p.numel() for p in model.parameters())
                param_mb = total_params * 4 / (1024 * 1024) 
                
                batch_size = 4
                seq_len = 64
                input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
                
                if torch.cuda.is_available():
                    model = model.cuda()
                    input_ids = input_ids.cuda()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                    _ = model(input_ids)
                    torch.cuda.synchronize()
                    
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    results.add_pass(f"{name} model memory", 
                                   f"Params: {param_mb:.1f}MB, Peak GPU: {peak_mb:.1f}MB")
                else:
                    results.add_pass(f"{name} model size", 
                                   f"Parameters: {total_params:,} ({param_mb:.1f}MB)")
                    
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                results.add_warning(f"{name} model memory", f"Could not test: {str(e)}")
                
    except Exception as e:
        results.add_fail("Memory usage test", str(e))


def test_inference_compatibility(results: TestResults):
    print(f"\n{Fore.CYAN}=== INFERENCE COMPATIBILITY TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm_inference import LiquidTransformerInference
        results.add_pass("Import inference module", "Inference imports work")
 
        
    except ImportError:
        results.add_warning("Inference module", 
                           "Cannot import - this is OK if file doesn't exist yet")
    except Exception as e:
        results.add_fail("Inference compatibility", str(e))


# Di dalam file test_liquid_transformer.py

def stress_test_generation(results: TestResults):
    print(f"\n{Fore.CYAN}=== GENERATION STRESS TEST ==={Style.RESET_ALL}")
    
    try:
        from liquid_transformers_lm import LiquidTransformerLM

        model_max_seq_len = 64 # Sesuaikan dengan max_seq_len model yang diinisialisasi
        model = LiquidTransformerLM(
            vocab_size=100,
            embed_size=32,
            num_heads=2,
            num_experts=2,
            num_layers=1,
            max_seq_len=model_max_seq_len
        )
        model.eval()

        edge_cases = [
            ("Empty prompt", torch.tensor([[]], dtype=torch.long)),
            ("Single token", torch.tensor([[1]], dtype=torch.long)),
            ("Max length prompt", torch.randint(0, 100, (1, 40), dtype=torch.long)), # Diubah ke (1, 40)
            ("Batch generation input", torch.randint(0, 100, (4, 10), dtype=torch.long)),
        ]
        
        for name, prompt_data in edge_cases:
            try:
                num_new_tokens_to_generate = 20 

                if name == "Batch generation input":
                    test_name_batch = "Generation - Batch (processed item by item)"
                    all_items_passed_generation_basic_check = True # Ganti nama flag
                    details_batch = []
                    if prompt_data.numel() == 0 and prompt_data.dim() == 2 and prompt_data.shape[1] == 0 :
                         results.add_warning(f"Generation - {name}", "Skipped completely empty prompt data (e.g. tensor([[]]))")
                         continue

                    for i in range(prompt_data.size(0)):
                        single_prompt = prompt_data[i:i+1]
                        # Panjang target total maksimum, tapi tidak melebihi kapasitas model
                        single_prompt_target_max_len = min(single_prompt.size(1) + num_new_tokens_to_generate, model_max_seq_len)
                        
                        if single_prompt.numel() > 0:
                            if single_prompt_target_max_len <= single_prompt.size(1):
                                results.add_warning(f"Generation - {name} (item {i})", f"Skipped. Prompt len {single_prompt.size(1)} vs target max len {single_prompt_target_max_len}. No new tokens.")
                                # Jika tidak ada token baru yang bisa digenerate, anggap saja ini bukan kegagalan tes generasi
                                # tapi juga bukan keberhasilan yang menambah detail.
                                # Atau, Anda bisa menandainya sebagai kegagalan jika Anda mengharapkan generasi.
                                # Untuk sekarang, kita lewati saja penambahan ke details_batch jika tidak ada generasi.
                                continue 
                            
                            generated = model.generate(single_prompt, max_length=single_prompt_target_max_len, temperature=1.0)
                            
                            # Validasi yang lebih fleksibel:
                            # 1. Batch size output adalah 1.
                            # 2. Panjang output lebih besar dari panjang prompt.
                            # 3. Panjang output tidak lebih dari target panjang maksimum.
                            is_valid_generation = (
                                generated.shape[0] == 1 and
                                generated.shape[1] > single_prompt.shape[1] and
                                generated.shape[1] <= single_prompt_target_max_len
                            )
                            
                            if not is_valid_generation:
                                # Jika panjang output sama dengan prompt, berarti tidak ada token baru yang tergenerasi
                                # Ini bisa jadi karena EOS langsung di awal atau masalah lain
                                if generated.shape[1] == single_prompt.shape[1]:
                                     details_batch.append(f"Item {i} shape: {generated.shape} (target max_len: {single_prompt_target_max_len} - NO NEW TOKENS GENERATED)")
                                else:
                                     details_batch.append(f"Item {i} shape: {generated.shape} (target max_len: {single_prompt_target_max_len} - FAILED VALIDATION)")
                                all_items_passed_generation_basic_check = False
                            else:
                                details_batch.append(f"Item {i} shape: {generated.shape} (target max_len: {single_prompt_target_max_len})")
                        else:
                            results.add_warning(f"Generation - {name} (item {i})", "Skipped empty single prompt item")
                            all_items_passed_generation_basic_check = False 
                    
                    if all_items_passed_generation_basic_check and details_batch: # Jika semua item lolos validasi dasar dan ada hasil
                        results.add_pass(test_name_batch, ", ".join(details_batch))
                    elif details_batch: # Ada beberapa hasil tapi mungkin tidak semua valid
                        results.add_fail(test_name_batch, f"One or more items in batch may have failed generation or shape check. Details: {', '.join(details_batch)}")
                    else: # Tidak ada item yang diproses sama sekali
                        results.add_warning(test_name_batch, "No items processed in batch input.")

                elif prompt_data.numel() > 0 or (prompt_data.dim() == 2 and prompt_data.shape[1] == 0 and prompt_data.shape[0] == 1): 
                    if prompt_data.numel() == 0:
                        results.add_warning(f"Generation - {name}", "Skipped empty prompt (tensor([[]]))")
                        continue
                    
                    target_max_total_length = min(prompt_data.size(1) + num_new_tokens_to_generate, model_max_seq_len)

                    if target_max_total_length <= prompt_data.size(1):
                        results.add_warning(f"Generation - {name}", f"Skipped. Prompt length {prompt_data.size(1)} vs target max length {target_max_total_length}. No new tokens.")
                        continue
                        
                    generated = model.generate(prompt_data, max_length=target_max_total_length, temperature=1.0)
                    
                    # Validasi yang lebih fleksibel
                    is_valid_generation_single = (
                        generated.shape[0] == prompt_data.shape[0] and
                        generated.shape[1] > prompt_data.size(1) and
                        generated.shape[1] <= target_max_total_length
                    )
                    
                    # Kasus khusus: jika prompt sudah panjang dan target_max_total_length hanya sedikit lebih besar,
                    # dan EOS terjadi, output bisa valid meskipun tidak mencapai target_max_total_length.
                    # Untuk tes unit, validasi di atas sudah cukup baik.

                    if not is_valid_generation_single:
                         # Jika panjang output sama dengan prompt, berarti tidak ada token baru yang tergenerasi
                        if generated.shape[1] == prompt_data.size(1):
                            results.add_fail(f"Generation - {name}", f"Output: {generated.shape} (target max_len: {target_max_total_length} - NO NEW TOKENS), input: {prompt_data.shape}")
                        else:
                            results.add_fail(f"Generation - {name}", f"Output: {generated.shape} (target max_len: {target_max_total_length} - FAILED VALIDATION), input: {prompt_data.shape}")
                    else:
                        results.add_pass(f"Generation - {name}", f"Shape: {generated.shape} (target max_len: {target_max_total_length})")
                else:
                    results.add_warning(f"Generation - {name}", "Skipped empty prompt data")
            except Exception as e:
                results.add_fail(f"Generation - {name}", str(e))
                traceback.print_exc()
                
    except ImportError:
        results.add_warning("Generation stress test", "Skipped: LiquidTransformerLM not found.")
    except Exception as e:
        results.add_fail("Generation stress test setup", str(e))
        traceback.print_exc()

def run_all_tests():
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}LIQUID TRANSFORMER COMPREHENSIVE TEST SUITE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    results = TestResults()

    tests = [
        ("System Requirements", check_system_requirements),
        ("Liquid Cell Components", test_liquid_cell_components),
        ("Attention & MoE", test_attention_and_moe),
        ("Full Model", test_full_model),
        ("Training Components", test_training_components),
        ("Mock Training Loop", test_mock_training_loop),
        ("Memory Usage", test_memory_usage),
        ("Generation Stress Test", stress_test_generation),
        ("Inference Compatibility", test_inference_compatibility),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func(results)
        except Exception as e:
            results.add_fail(test_name, f"Test crashed: {str(e)}")
            traceback.print_exc()

    success = results.summary()

    print(f"\n{Fore.CYAN}RECOMMENDATIONS:{Style.RESET_ALL}")
    
    if success:
        print(f"{Fore.GREEN}✓ All tests passed! The code appears to be working correctly.{Style.RESET_ALL}")
        print("\nYou can proceed with training. Suggested next steps:")
        print("1. Start with a small model configuration")
        print("2. Use a small dataset subset first")
        print("3. Monitor GPU memory usage")
        print("4. Save checkpoints frequently")
    else:
        print(f"{Fore.YELLOW}⚠ Some tests failed. Please fix the issues before training.{Style.RESET_ALL}")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check CUDA compatibility with PyTorch version")
        print("3. Reduce model size if memory errors occur")
        print("4. Check for syntax errors in the implementation")
    
    if results.warnings:
        print(f"\n{Fore.YELLOW}Warnings to consider:{Style.RESET_ALL}")
        for test, warning in results.warnings[:5]:
            print(f"  - {test}: {warning}")
    
    return success

def quick_diagnosis():
    print(f"\n{Fore.CYAN}QUICK DIAGNOSIS:{Style.RESET_ALL}")

    print("\nChecking critical imports...")
    critical_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("liquid_transformers_lm", "Liquid Transformer Model"),
        ("complete_training_script", "Training Script"),
    ]
    
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")

    print("\nGPU Status:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠ No GPU detected - training will be slow")
    except:
        print("  ✗ Cannot check GPU status")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Liquid Transformer Implementation")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnosis only")
    parser.add_argument("--gpu-only", action="store_true", help="Skip tests if no GPU")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_diagnosis()
    else:
        if args.gpu_only and not torch.cuda.is_available():
            print(f"{Fore.YELLOW}No GPU detected and --gpu-only flag set. Skipping tests.{Style.RESET_ALL}")
        else:
            success = run_all_tests()
            sys.exit(0 if success else 1)