import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json

from lfm.liquid_transformers_lm import (
    LiquidTransformerLM,
    AdaptiveLearningRateScheduler,
    curriculum_sequence_length
)


class CurriculumLanguageModelingDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        min_length: int = 64,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.min_length = min_length
        self.max_length = max_length
     
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) >= min_length:
                self.examples.append(tokens)
        
        # Sort by length
        self.examples.sort(key=len)
        logger.info(f"Dataset size: {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def get_batch_with_length(self, indices: List[int], max_length: int) -> Dict[str, torch.Tensor]:
        batch_tokens = []
        attention_masks = []
        
        for idx in indices:
            tokens = self.examples[idx][:max_length]
            padding_length = max_length - len(tokens)
            
            # Pad tokens
            padded_tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(tokens) + [0] * padding_length
            
            batch_tokens.append(padded_tokens)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
        }


class LiquidTransformerTrainer:
    def __init__(
        self,
        model: LiquidTransformerLM,
        tokenizer: AutoTokenizer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Adaptive LR scheduler
        self.scheduler = AdaptiveLearningRateScheduler(
            self.optimizer,
            base_lr=learning_rate
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.mixed_precision = mixed_precision
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'sequence_length': [],
            'liquid_stability': [],
            'expert_usage': []
        }
        
        if use_wandb:
            wandb.init(
                project="complete-liquid-transformer",
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "model_params": sum(p.numel() for p in model.parameters())
                }
            )
        
        self.use_wandb = use_wandb
        logger.info("Trainer initialized with all features")
    
    def compute_loss_with_aux(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        liquid_states: Optional[List[Dict[str, torch.Tensor]]] = None,
        time_step: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:

        # Forward pass
        logits, new_liquid_states, aux_losses = self.model(
            input_ids[:, :-1],
            attention_mask[:, :-1],
            liquid_states,
            time_step
        )
        
        # Language modeling loss
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Total loss with auxiliary losses
        total_loss = lm_loss
        for loss_name, loss_value in aux_losses.items():
            if 'loss' in loss_name:
                total_loss = total_loss + loss_value
        
        return total_loss, new_liquid_states, aux_losses
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        liquid_states: Optional[List[Dict[str, torch.Tensor]]],
        global_step: int,
        max_steps: int
    ) -> Tuple[float, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:

        self.model.train()
        
        # Time step for temporal dynamics
        time_step = global_step / max_steps
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Mixed precision context
        with torch.cuda.amp.autocast() if self.mixed_precision else torch.no_grad():
            loss, liquid_states, aux_losses = self.compute_loss_with_aux(
                input_ids, attention_mask, liquid_states, time_step
            )
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps, liquid_states, aux_losses
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        batch_size: int = 16,
        max_length: int = 512
    ) -> Dict[str, float]:

        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        all_aux_losses = {}
        
        with torch.no_grad():
            for i in range(0, len(eval_dataset), batch_size):
                indices = list(range(i, min(i + batch_size, len(eval_dataset))))
                batch = eval_dataset.get_batch_with_length(indices, max_length)
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                loss, _, aux_losses = self.compute_loss_with_aux(
                    input_ids, attention_mask
                )
                
                num_tokens = (attention_mask == 1).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Aggregate auxiliary metrics
                for k, v in aux_losses.items():
                    if k not in all_aux_losses:
                        all_aux_losses[k] = []
                    all_aux_losses[k].append(v.item() if torch.is_tensor(v) else v)
        
        # Compute averages
        metrics = {
            'eval_loss': total_loss / total_tokens,
            'eval_perplexity': np.exp(total_loss / total_tokens)
        }
        
        for k, v in all_aux_losses.items():
            metrics[f'eval_{k}'] = np.mean(v)
        
        return metrics
    
    def measure_adaptivity(self, test_sequences: List[torch.Tensor]) -> float:
        self.model.eval()
        adaptivity_scores = []
        
        with torch.no_grad():
            for seq in test_sequences:
                seq = seq.to(self.device)
                
                # Original sequence
                _, liquid_states1, _ = self.model(seq.unsqueeze(0))
                
                # Perturbed sequence
                noise = torch.randn_like(seq.float()) * 0.1
                perturbed = seq + noise.long()
                perturbed = torch.clamp(perturbed, 0, self.model.token_embedding.num_embeddings - 1)
                
                _, liquid_states2, _ = self.model(perturbed.unsqueeze(0))
                
                # Measure adaptation
                differences = []
                for ls1, ls2 in zip(liquid_states1, liquid_states2):
                    if ls1['combined'] is not None and ls2['combined'] is not None:
                        diff = torch.norm(ls2['combined'] - ls1['combined']) / torch.norm(ls1['combined'])
                        differences.append(diff.item())
                
                if differences:
                    adaptivity_scores.append(np.mean(differences))
        
        return np.mean(adaptivity_scores) if adaptivity_scores else 0.0
    
    def visualize_liquid_dynamics(self, sequence: torch.Tensor, save_path: str = "liquid_dynamics.png"):
        self.model.eval()
        
        with torch.no_grad():
            all_states = {'fast': [], 'medium': [], 'slow': [], 'combined': []}
            
            for t in range(1, len(sequence) + 1):
                partial_seq = sequence[:t].unsqueeze(0).to(self.device)
                _, liquid_states, _ = self.model(partial_seq)
                
                middle_layer = len(liquid_states) // 2
                for key in all_states:
                    if liquid_states[middle_layer][key] is not None:
                        state = liquid_states[middle_layer][key][0].cpu().numpy()
                        all_states[key].append(state)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ['Fast Dynamics', 'Medium Dynamics', 'Slow Dynamics', 'Combined Output']
        
        for idx, (key, title) in enumerate(zip(all_states.keys(), titles)):
            ax = axes[idx // 2, idx % 2]
            if all_states[key]:
                states_matrix = np.array(all_states[key]).T
                im = ax.imshow(states_matrix, aspect='auto', cmap='viridis')
                ax.set_title(title)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Hidden Dimension')
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved liquid dynamics visualization to {save_path}")
    
    def train(
        self,
        train_dataset: CurriculumLanguageModelingDataset,
        eval_dataset: Optional[CurriculumLanguageModelingDataset] = None,
        batch_size: int = 16,
        max_steps: int = 100000,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        log_steps: int = 10,
        gradient_clip: float = 1.0
    ):
        global_step = 0
        best_eval_loss = float('inf')
        accumulated_loss = 0.0
        liquid_states = None
        
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        while global_step < max_steps:
            current_seq_length = curriculum_sequence_length(
                global_step, 
                min_len=train_dataset.min_length,
                max_len=train_dataset.max_length
            )

            indices = np.random.choice(len(train_dataset), batch_size)
            batch = train_dataset.get_batch_with_length(indices, current_seq_length)

            loss, liquid_states, aux_losses = self.train_step(
                batch, liquid_states, global_step, max_steps
            )
            accumulated_loss += loss

            if (global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

                current_lr = self.scheduler.step(liquid_states)

                if global_step % log_steps == 0:
                    avg_loss = accumulated_loss / self.gradient_accumulation_steps
                    
                    metrics = {
                        "train_loss": avg_loss,
                        "learning_rate": current_lr,
                        "sequence_length": current_seq_length,
                        "global_step": global_step
                    }

                    for k, v in aux_losses.items():
                        metrics[f"train_{k}"] = v.item() if torch.is_tensor(v) else v

                    if liquid_states:
                        stabilities = []
                        for ls in liquid_states:
                            if ls['combined'] is not None:
                                stability = 1.0 / (1.0 + ls['combined'].var().item())
                                stabilities.append(stability)
                        if stabilities:
                            metrics["liquid_stability"] = np.mean(stabilities)

                    if self.use_wandb:
                        wandb.log(metrics)
                    
                    for k, v in metrics.items():
                        if k in self.metrics_history:
                            self.metrics_history[k].append(v)
                    
                    accumulated_loss = 0.0

            if eval_dataset and global_step % eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataset, batch_size, current_seq_length)

                if global_step % (eval_steps * 5) == 0:
                    test_seqs = [eval_dataset.get_batch_with_length([i], 128)["input_ids"][0] 
                                for i in range(10)]
                    adaptivity = self.measure_adaptivity(test_seqs)
                    eval_metrics['adaptivity_score'] = adaptivity
                
                if self.use_wandb:
                    wandb.log(eval_metrics)
                
                logger.info(f"Step {global_step}: eval_loss = {eval_metrics['eval_loss']:.4f}, "
                          f"eval_perplexity = {eval_metrics['eval_perplexity']:.2f}")
                
                if eval_metrics['eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_metrics['eval_loss']
                    self.save_checkpoint(global_step, is_best=True)

            if global_step % save_steps == 0:
                self.save_checkpoint(global_step)

                sample_seq = train_dataset.get_batch_with_length([0], 128)["input_ids"][0]
                viz_path = self.checkpoint_dir / f"liquid_dynamics_step_{global_step}.png"
                self.visualize_liquid_dynamics(sample_seq, str(viz_path))

            if global_step % (save_steps * 2) == 0:
                self.generate_and_log_sample(global_step)
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "seq_len": current_seq_length
            })
        
        progress_bar.close()
        logger.info("Training completed!")

        self.save_checkpoint(global_step, is_final=True)
        
        self.save_metrics_history()
    
    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        elif is_final:
            checkpoint_path = self.checkpoint_dir / "final_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "config": {
                "vocab_size": self.model.token_embedding.num_embeddings,
                "embed_size": self.model.embed_size,
                "num_layers": self.model.num_layers
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics_history = checkpoint.get("metrics_history", self.metrics_history)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get("step", 0)
    
    def save_metrics_history(self):
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
    
    def generate_and_log_sample(self, step: int):
        self.model.eval()
        
        prompts = [
            "The future of artificial intelligence",
            "In the realm of quantum mechanics",
            "The fundamental principles of",
            "Recent advances in machine learning"
        ]
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_length=150,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            logger.info(f"Step {step} - Generated sample:\n{generated_text}\n")
            
            if self.use_wandb:
                wandb.log({
                    f"generated_text_{prompt[:20]}": generated_text,
                    "global_step": step
                })


def main():
    model_config = {
        "vocab_size": 30522,
        "embed_size": 768,
        "num_heads": 12,
        "num_experts": 4,
        "num_layers": 6,
        "max_seq_len": 512,
        "dropout": 0.1
    }
    
    training_config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_steps": 100000,
        "eval_steps": 1000,
        "save_steps": 5000,
        "mixed_precision": True
    }

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    logger.info("Loading dataset...")
    dataset = load_dataset("neuralwork/arxiver", split="train", streaming=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= 50000:
            break
        text = f"{item['title']}. {item['abstract']}"
        texts.append(text)

    split_idx = int(0.9 * len(texts))
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    logger.info("Creating datasets...")
    train_dataset = CurriculumLanguageModelingDataset(
        tokenizer, train_texts, min_length=64, max_length=512
    )
    eval_dataset = CurriculumLanguageModelingDataset(
        tokenizer, eval_texts, min_length=64, max_length=512
    )
    
    logger.info("Initializing model...")
    model =LiquidTransformerLM(**model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = LiquidTransformerTrainer(
        model=model,
        tokenizer=tokenizer,
        **training_config
    )
    
    logger.info("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=training_config["batch_size"],
        max_steps=training_config["max_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"]
    )
    
    logger.info("Final evaluation...")
    final_metrics = trainer.evaluate(eval_dataset)
    logger.info(f"Final metrics: {final_metrics}")
    
    test_sequences = [eval_dataset.get_batch_with_length([i], 128)["input_ids"][0] for i in range(20)]
    final_adaptivity = trainer.measure_adaptivity(test_sequences)
    logger.info(f"Final adaptivity score: {final_adaptivity:.4f}")

if __name__ == "__main__":
    main()