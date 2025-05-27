import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
import json
import numpy as np
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
from loguru import logger
import time

from lfm.liquid_transformers_lm import LiquidTransformerLM


class LiquidTransformerInference:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "bert-base-uncased",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        logger.info("Inference engine initialized")
    
    def load_model(self, checkpoint_path: str) -> LiquidTransformerLM:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        config = checkpoint.get("config", {
            "vocab_size": 30522,
            "embed_size": 768,
            "num_heads": 12,
            "num_experts": 4,
            "num_layers": 6,
            "max_seq_len": 512
        })
        
        model = LiquidTransformerLM(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Model was trained for {checkpoint.get('step', 'unknown')} steps")
        
        return model
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        early_stopping: bool = True
    ) -> List[str]:

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)
        
        generated_sequences = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                if do_sample:
                    output_ids = self._generate_sample(
                        input_ids,
                        attention_mask,
                        max_length,
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty,
                        early_stopping
                    )
                else:
                    output_ids = self._generate_greedy(
                        input_ids,
                        attention_mask,
                        max_length,
                        early_stopping
                    )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def _generate_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        early_stopping: bool
    ) -> torch.Tensor:

        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        generated = input_ids.clone()
        liquid_states = None
        past_tokens = set(input_ids[0].tolist())
        eos_token_id = self.tokenizer.eos_token_id or 0

        for step in range(max_length - cur_len):
            # Time step for adaptive dynamics
            time_step = step / (max_length - cur_len)
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
                logits, liquid_states, _ = self.model(
                    generated,
                    attention_mask,
                    liquid_states,
                    time_step
                )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            for token_id in past_tokens:
                next_token_logits[:, token_id] /= repetition_penalty
            
            # Filter logits
            filtered_logits = self._top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )
            
            # Sample
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update
            generated = torch.cat([generated, next_token], dim=1)
            past_tokens.add(next_token.item())
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device)
                ], dim=1)

            if early_stopping and next_token.item() == eos_token_id:
                break
        
        return generated
    
    def _generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_length: int,
        early_stopping: bool
    ) -> torch.Tensor:

        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        
        generated = input_ids.clone()
        liquid_states = None
        eos_token_id = self.tokenizer.eos_token_id or 0
        
        for step in range(max_length - cur_len):
            time_step = step / (max_length - cur_len)
            
            # Forward pass
            logits, liquid_states, _ = self.model(
                generated,
                attention_mask,
                liquid_states,
                time_step
            )
            
            # Greedy selection
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Update
            generated = torch.cat([generated, next_token], dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device)
                ], dim=1)
            
            # Early stopping
            if early_stopping and next_token.item() == eos_token_id:
                break
        
        return generated
    
    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:

        assert logits.dim() == 2  # batch_size x vocab_size
        
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def beam_search(
        self,
        prompt: str,
        beam_width: int = 5,
        max_length: int = 100,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> List[Tuple[str, float]]:

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]

        beams = [(input_ids, 0.0, None)]  # (sequence, score, liquid_states)
        eos_token_id = self.tokenizer.eos_token_id or 0
        
        for step in range(max_length - input_ids.size(1)):
            all_candidates = []
            
            for seq, score, liquid_states in beams:
                if seq[0, -1].item() == eos_token_id and early_stopping:
                    all_candidates.append((seq, score, liquid_states))
                    continue

                with torch.no_grad():
                    logits, new_liquid_states, _ = self.model(
                        seq, liquid_states=liquid_states
                    )

                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    new_seq = torch.cat([seq, topk_indices[:, i:i+1]], dim=1)
                    new_score = score + topk_log_probs[:, i].item()

                    normalized_score = new_score / (new_seq.size(1) ** length_penalty)
                    
                    all_candidates.append((new_seq, normalized_score, new_liquid_states))

            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        results = []
        for seq, score, _ in beams:
            text = self.tokenizer.decode(seq[0], skip_special_tokens=True)
            results.append((text, score))
        
        return results
    
    def analyze_liquid_dynamics(
        self,
        prompt: str,
        max_length: int = 50,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
  
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        all_states = {
            'fast': [],
            'medium': [],
            'slow': [],
            'combined': [],
            'tokens': []
        }
        
        generated = input_ids.clone()
        liquid_states = None
        
        with torch.no_grad():
            for step in range(max_length):
                logits, liquid_states, _ = self.model(generated, liquid_states=liquid_states)
                
                middle_layer = len(liquid_states) // 2
                for key in ['fast', 'medium', 'slow', 'combined']:
                    if liquid_states[middle_layer][key] is not None:
                        state = liquid_states[middle_layer][key][0].cpu().numpy()
                        all_states[key].append(state)

                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                all_states['tokens'].append(self.tokenizer.decode(next_token[0]))

        if save_path:
            self._plot_liquid_dynamics(all_states, save_path)
        
        return all_states
    
    def _plot_liquid_dynamics(self, states: Dict[str, List], save_path: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (key, title) in enumerate([
            ('fast', 'Fast Dynamics'),
            ('medium', 'Medium Dynamics'),
            ('slow', 'Slow Dynamics'),
            ('combined', 'Combined Output')
        ]):
            ax = axes[idx // 2, idx % 2]
            
            if states[key]:
                data = np.array(states[key]).T
                im = ax.imshow(data, aspect='auto', cmap='viridis')
                ax.set_title(title)
                ax.set_xlabel('Generation Step')
                ax.set_ylabel('Hidden Dimension')

                plt.colorbar(im, ax=ax)
                
                if len(states['tokens']) < 30:
                    ax.set_xticks(range(len(states['tokens'])))
                    ax.set_xticklabels(states['tokens'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dynamics plot to {save_path}")
    
    def interactive_generation(self):
        print("\n" + "="*50)
        print("Liquid Transformer Interactive Generation")
        print("="*50)
        print("Commands:")
        print("  /temp <value>    - Set temperature (default: 1.0)")
        print("  /length <value>  - Set max length (default: 100)")
        print("  /beam            - Use beam search")
        print("  /analyze         - Analyze liquid dynamics")
        print("  /quit            - Exit")
        print("="*50 + "\n")
        
        settings = {
            'temperature': 1.0,
            'max_length': 100,
            'top_k': 50,
            'top_p': 0.95
        }
        
        while True:
            prompt = input("\nEnter prompt (or command): ").strip()
            
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0]
                
                if cmd == '/quit':
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    settings['temperature'] = float(parts[1])
                    print(f"Temperature set to {settings['temperature']}")
                elif cmd == '/length' and len(parts) > 1:
                    settings['max_length'] = int(parts[1])
                    print(f"Max length set to {settings['max_length']}")
                elif cmd == '/beam':
                    real_prompt = input("Enter prompt for beam search: ")
                    results = self.beam_search(real_prompt, beam_width=5)
                    print("\nBeam Search Results:")
                    for i, (text, score) in enumerate(results):
                        print(f"\n--- Beam {i+1} (score: {score:.4f}) ---")
                        print(text)
                elif cmd == '/analyze':
                    real_prompt = input("Enter prompt to analyze: ")
                    print("Analyzing liquid dynamics...")
                    self.analyze_liquid_dynamics(
                        real_prompt,
                        max_length=30,
                        save_path="liquid_dynamics_analysis.png"
                    )
                    print("Analysis saved to liquid_dynamics_analysis.png")
                else:
                    print("Unknown command")
                continue
            
            print("\nGenerating...")
            start_time = time.time()
            
            generated = self.generate(
                prompt,
                max_length=settings['max_length'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p']
            )
            
            elapsed = time.time() - start_time
            
            print(f"\nGenerated in {elapsed:.2f}s:")
            print("-" * 50)
            print(generated[0])
            print("-" * 50)
    
    def benchmark_generation_speed(
        self,
        prompt: str = "The future of artificial intelligence",
        lengths: List[int] = [50, 100, 200],
        batch_sizes: List[int] = [1, 4, 8]
    ):
        print("\nBenchmarking Generation Speed")
        print("="*50)
        
        results = {}
        
        for batch_size in batch_sizes:
            for length in lengths:
                prompts = [prompt] * batch_size
                
                start = time.time()
                _ = self.generate(
                    prompts[0],
                    max_length=length,
                    temperature=1.0
                )
                elapsed = time.time() - start
                
                tokens_per_second = length / elapsed
                results[f"batch_{batch_size}_len_{length}"] = {
                    'time': elapsed,
                    'tokens_per_second': tokens_per_second
                }
                
                print(f"Batch {batch_size}, Length {length}: "
                      f"{elapsed:.2f}s ({tokens_per_second:.1f} tokens/s)")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Liquid Transformer Inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Generation prompt")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive mode")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze liquid dynamics")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark")
    parser.add_argument("--max-length", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p filtering")
    parser.add_argument("--beam-search", action="store_true",
                        help="Use beam search")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width for beam search")
    
    args = parser.parse_args()
    
    engine = LiquidTransformerInference(args.model_path)
    
    if args.interactive:
        engine.interactive_generation()
    elif args.benchmark:
        engine.benchmark_generation_speed()
    elif args.analyze and args.prompt:
        print("Analyzing liquid dynamics...")
        engine.analyze_liquid_dynamics(
            args.prompt,
            max_length=args.max_length,
            save_path="liquid_dynamics.png"
        )
        print("Analysis saved to liquid_dynamics.png")
    elif args.prompt:
        if args.beam_search:
            print("Using beam search...")
            results = engine.beam_search(
                args.prompt,
                beam_width=args.beam_width,
                max_length=args.max_length
            )
            for i, (text, score) in enumerate(results):
                print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
                print(text)
        else:
            print("Generating...")
            generated = engine.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            print("\nGenerated:")
            print("-" * 50)
            print(generated[0])
            print("-" * 50)
    else:
        print("Please provide --prompt or use --interactive mode")


if __name__ == "__main__":
    main()