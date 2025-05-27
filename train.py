import os
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from loguru import logger
import wandb
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from lfm.liquid_transformers_moe import LiquidTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger.add(
    "training.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

@dataclass
class TrainingConfig:
    embed_size: int = 768
    num_heads: int = 8
    num_experts: int = 4
    expert_size: int = 768
    num_layers: int = 6

    batch_size: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 100000
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    max_length: int = 512
    vocab_size: int = 30522

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    seed: int = 42

    wand_project: str = "liquid-models"
    checkpoint_dir: str = "checkpoints"
    checkpoint_steps: int = 1000
    log_steps: int = 10

class Dataset(IterableDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("neuralwork/arxiver", split="train", streaming=True)
        logger.info(f"initialized dataset")

    def preprocess_text(self, text: str) -> str:
        return text.strip().replace('\n', ' ')
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iterator = iter(self.dataset)
        while True:
            try:
                item = next(iterator)
                text = f"Title: {self.preprocess_text(item['title'])} Abstract: {self.preprocess_text(item['abstract'])}"

                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                yield {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0]
                }
            
            except StopIteration:
                iterator = iter(self.dataset)
                continue

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer: AutoTokenizer
    ):
        self.model = model.to(config.device)
        self.config = config
        self.tokenizer = tokenizer

        self.model.hidden_state = torch.zeros(
            config.batch_size,
            config.embed_size,
            device=config.device
        )

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_size
        ).to(config.device)

        self.optimizer = AdamW(
            list(model.parameters()) + list(self.embedding.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps
        )

        wandb.init(project=config.wand_project, config=vars(config))
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        logger.info("trainer initialized")

    def train_steps(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> float:
        try:
            self.model.train()

            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)

            embedded_input = self.embedding(input_ids)
            embedded_input = embedded_input.unsqueeze(0)

            if self.model.hidden_state.size(0) != embedded_input.size(1):
                self.model.hidden_state = self.model.hidden_state.new_zeros(
                    embedded_input.size(1),
                    self.config.embed_size
                )

            outputs = self.model(embedded_input)

            loss = nn.MSELoss()(outputs, embedded_input)

            loss.backward()
            torch.nn.utils.clip_grad_norm(
                list(self.model.parameters()) + list(self.embedding.parameters()),
                self.config.max_grad_norm
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
        
        except Exception as e:
            logger.error(f"error in train_step: {str(e)}")
            raise

    def save_checkpoint(
        self,
        step: int,
        loss: Optional[float] = None
    ):
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "embedding_state_dict": self.embedding.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss if loss is not None else float('inf'),
            "config": self.config
        }

        path = Path(self.config.checkpoint_dir)
        checkpoint_path = path / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"saved checkpoint at step {step} to {checkpoint_path}")

    def train(
        self,
        train_dataset: Dataset
    ):
        logger.info("starting training")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )

        global_step = 0
        running_loss = 0.0
        current_loss = None

        progress_bar = tqdm(total=self.config.max_steps, desc="training")

        try:
            for batch in train_loader:
                if global_step >= self.config.max_steps:
                    break
                    
                current_loss = self.train_steps(batch)
                running_loss += current_loss
                global_step += 1

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "step": global_step
                })

                if global_step % self.config.checkpoint_steps == 0:
                    self.save_checkpoint(global_step, current_loss)

                self.scheduler.step()

        except KeyboardInterrupt:
            logger.info("training intterupted by user")
            self.save_checkpoint(global_step, current_loss)
        except Exception as e:
            logger.error(f"training erro> {str(e)}")
            self.save_checkpoint(global_step, current_loss)
            raise
        finally:
            progress_bar.close()
            self.save_checkpoint(global_step, current_loss)
            logger.info(f"training completed after {global_step} steps")

def main():
    try:
        config = TrainingConfig()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        train_dataset = Dataset(
            tokenizer=tokenizer,
            max_length=config.max_length
        )

        model = LiquidTransformer(
            embed_size=config.embed_size,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            expert_size=config.expert_size,
            num_layers=config.num_layers
        )

        trainer = Trainer(model, config, tokenizer)

        trainer.train(train_dataset)

    except Exception as e:
        logger.error(f"training failed with error: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()