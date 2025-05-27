import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np
from loguru import logger
import math

class ODESolver:
    @staticmethod
    def odeint(func, y0, t, method='euler'):
        dt = t[1] - t[0]
        y = y0
        for i in range(len(t) - 1):
            if method == 'euler':
                y = y + dt * func(t[i], y)
            elif method == 'rk4':
                k1 = func(t[i], y)
                k2 = func(t[i] + dt/2, y + dt*k1/2)
                k3 = func(t[i] + dt/2, y + dt*k2/2)
                k4 = func(t[i] + dt, y + dt*k3)
                y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        return y


class LiquidCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_constant: float = 1.0,
        ode_steps: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        self.ode_steps = ode_steps
        
        self.ode_func_net = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.complexity_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.time_modulation = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def ode_func(self, t: float, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h_x = torch.cat([h, x], dim=-1)

        dh_dt = self.ode_func_net(h_x)
        
        t_tensor = torch.tensor([t], device=h.device).expand(h.size(0), 1)
        time_mod = self.time_modulation(t_tensor)
        dh_dt = dh_dt * time_mod
        
        return dh_dt / self.time_constant
    
    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor,
        adapt_steps: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        if adapt_steps is None:
            complexity = self.complexity_gate(x)
            adapt_steps = 5 + int(complexity.mean() * self.ode_steps)

        t = torch.linspace(0, 1, adapt_steps).to(device)

        ode_solver = ODESolver()
        h_trajectory = h.unsqueeze(0)
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dh = self.ode_func(t[i-1].item(), h_trajectory[-1], x)
            h_new = h_trajectory[-1] + dt * dh
            h_trajectory = torch.cat([h_trajectory, h_new.unsqueeze(0)], dim=0)

        h_final = h_trajectory[-1]

        h_final = self.layer_norm(h_final)
        h_final = self.dropout(h_final)
        
        return h_final


class HierarchicalLiquidCell(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        # Fast dynamics (token-level)
        self.fast_liquid = LiquidCell(
            hidden_size, hidden_size, 
            time_constant=0.1, ode_steps=5, dropout=dropout
        )
        
        # Medium dynamics (sentence-level)
        self.medium_liquid = LiquidCell(
            hidden_size, hidden_size,
            time_constant=1.0, ode_steps=10, dropout=dropout
        )
        
        # Slow dynamics (document-level)
        self.slow_liquid = LiquidCell(
            hidden_size, hidden_size,
            time_constant=10.0, ode_steps=15, dropout=dropout
        )
        
        # Mixing weights
        self.mix_weights = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        fast_h: torch.Tensor,
        medium_h: torch.Tensor,
        slow_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        fast_h = self.fast_liquid(x, fast_h)
        medium_h = self.medium_liquid(x, medium_h)
        slow_h = self.slow_liquid(x, slow_h)
        
        # Compute mixing weights based on input
        weights = self.mix_weights(x)
        
        # Combine scales
        combined = (
            weights[:, 0:1] * fast_h +
            weights[:, 1:2] * medium_h +
            weights[:, 2:3] * slow_h
        )
        
        return combined, fast_h, medium_h, slow_h


class TimeAwareAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Learnable time encoding
        self.time_encoding = nn.Sequential(
            nn.Linear(1, embed_size // 2),
            nn.GELU(),
            nn.Linear(embed_size // 2, embed_size)
        )
        
        # Relative position encoding
        self.rel_pos_encoding = nn.Parameter(
            torch.randn(1, 512, embed_size) * 0.02
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        time_step: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size, seq_len, embed_size = x.shape
        
        # Add relative position encoding
        rel_pos = self.rel_pos_encoding[:, :seq_len, :]
        x = x + rel_pos
        
        # Add time encoding if provided
        if time_step is not None:
            time_tensor = torch.tensor([[time_step]], device=x.device)
            time_enc = self.time_encoding(time_tensor)
            x = x + time_enc.unsqueeze(1)
        
        # Apply attention
        attn_out, attn_weights = self.attention(
            x, x, x, key_padding_mask=attention_mask
        )
        
        return attn_out, attn_weights


class AdaptiveMixtureOfExperts(nn.Module):
    def __init__(
        self, 
        num_experts: int, 
        input_dim: int, 
        hidden_dim: int,
        liquid_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        
        # Expert networks with different capacities
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim * (i + 1)),
                nn.LayerNorm(hidden_dim * (i + 1)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * (i + 1), input_dim)
            )
            for i in range(num_experts)
        ])
        
        # Adaptive routing that considers liquid state
        self.router = nn.Sequential(
            nn.Linear(input_dim + liquid_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Uncertainty estimator for dynamic top-k
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Load balancing loss
        self.load_balance_alpha = 0.01
        
    def forward(
        self, 
        x: torch.Tensor, 
        liquid_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
 
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        liquid_flat = liquid_state.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, liquid_state.size(-1))
        
        # Compute routing probabilities
        router_input = torch.cat([x_flat, liquid_flat], dim=-1)
        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Estimate uncertainty for dynamic top-k
        uncertainty = self.uncertainty_net(x_flat).squeeze(-1)
        k = torch.clamp(2 + uncertainty * (self.num_experts - 2), 2, self.num_experts).int()
        
        k = 2
        topk_probs, topk_indices = router_probs.topk(k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        expert_loads = torch.zeros(self.num_experts, device=x.device)
        
        for i in range(k):
            for expert_id in range(self.num_experts):
                mask = (topk_indices[:, i] == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += topk_probs[mask, i:i+1] * expert_output
                    expert_loads[expert_id] += mask.float().sum()
        
        output = output.view(batch_size, seq_len, dim)
        
        # Compute load balancing loss
        expert_loads = expert_loads / expert_loads.sum()
        uniform_load = 1.0 / self.num_experts
        load_balance_loss = self.load_balance_alpha * F.mse_loss(
            expert_loads, 
            torch.full_like(expert_loads, uniform_load)
        )
        
        aux_losses = {
            'load_balance_loss': load_balance_loss,
            'expert_loads': expert_loads,
            'mean_uncertainty': uncertainty.mean()
        }
        
        return output, aux_losses


class LiquidTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        num_experts: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Time-aware attention
        self.attention = TimeAwareAttention(embed_size, num_heads, dropout)
        
        # Hierarchical liquid cell
        self.liquid_cell = HierarchicalLiquidCell(embed_size, dropout)
        
        # Adaptive MoE
        self.moe = AdaptiveMixtureOfExperts(
            num_experts, embed_size, embed_size * 4, embed_size, dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        liquid_states: Dict[str, torch.Tensor],
        time_step: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
   
        # Time-aware attention
        attn_out, attn_weights = self.attention(x, time_step, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Hierarchical liquid processing
        seq_repr = x.mean(dim=1)  # Aggregate sequence

        val_fast_h = liquid_states.get('fast')
        initial_fast_h = val_fast_h if torch.is_tensor(val_fast_h) else torch.zeros_like(seq_repr)
        
        val_medium_h = liquid_states.get('medium')
        initial_medium_h = val_medium_h if torch.is_tensor(val_medium_h) else torch.zeros_like(seq_repr)
        
        val_slow_h = liquid_states.get('slow')
        initial_slow_h = val_slow_h if torch.is_tensor(val_slow_h) else torch.zeros_like(seq_repr)

        liquid_out, fast_h_ret, medium_h_ret, slow_h_ret = self.liquid_cell(
            seq_repr,
            initial_fast_h,
            initial_medium_h,
            initial_slow_h
        )

        if liquid_out is None:
            liquid_out = torch.zeros_like(seq_repr) 

        new_liquid_states = {
            'fast': fast_h_ret,     
            'medium': medium_h_ret, 
            'slow': slow_h_ret,   
            'combined': liquid_out
        }
        
        # Expand liquid output to sequence length
        liquid_expanded = liquid_out.unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.norm2(x + liquid_expanded)
        
        # Adaptive MoE
        moe_out, aux_losses = self.moe(x, liquid_out)
        x = self.norm3(x + self.dropout(moe_out))
        
        return x, new_liquid_states, aux_losses


class LiquidTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        num_experts: int,
        num_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)
        
        # Learnable CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LiquidTransformerLayer(
                embed_size, num_heads, num_experts, dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        liquid_states: Optional[List[Dict[str, torch.Tensor]]] = None,
        current_time_step: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if liquid_states is None:
            liquid_states = [
                {'fast': None, 'medium': None, 'slow': None, 'combined': None}
                for _ in range(self.num_layers)
            ]
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        x = self.dropout(x)
        
        # Process through layers
        new_liquid_states = []
        all_aux_losses = {}
        
        for i, layer in enumerate(self.layers):
            x, layer_liquid_states, aux_losses = layer(
                x, liquid_states[i], current_time_step, attention_mask
            )
            new_liquid_states.append(layer_liquid_states)
            
            # Aggregate auxiliary losses
            for k, v in aux_losses.items():
                if k not in all_aux_losses:
                    all_aux_losses[k] = []
                all_aux_losses[k].append(v)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        # Final projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Average auxiliary losses across layers
        for k in all_aux_losses:
            all_aux_losses[k] = torch.stack(all_aux_losses[k]).mean()
        
        return logits, new_liquid_states, all_aux_losses
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2
    ) -> torch.Tensor:

        self.eval()
        liquid_states = None
        generated = input_ids.clone()
        past_tokens = set(input_ids[0].tolist())
        
        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # Adaptive time step
                time_step = step / max_length
                
                # Forward pass
                logits, liquid_states, _ = self.forward(
                    generated, 
                    liquid_states=liquid_states,
                    current_time_step=time_step
                )
                
                # Get last token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                for token in past_tokens:
                    next_token_logits[:, token] /= repetition_penalty
                
                # Apply top-k and top-p filtering
                filtered_logits = self._top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                
                # Sample
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Update generated and past tokens
                generated = torch.cat([generated, next_token], dim=1)
                past_tokens.add(next_token.item())
                
                # Early stopping on EOS token
                if self.token_embedding.num_embeddings > 0:
                    if next_token.item() == 0:
                        break
        
        return generated
    
    @staticmethod
    def _top_k_top_p_filtering(logits, top_k=50, top_p=0.95):

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Top-k filtering
        if top_k > 0:
            sorted_indices_to_remove[..., top_k:] = True
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        
        return logits


# Training utilities
class AdaptiveLearningRateScheduler:
    def __init__(self, optimizer, base_lr=1e-4, stability_window=100):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.stability_window = stability_window
        self.stability_history = []
        
    def step(self, liquid_states: List[Dict[str, torch.Tensor]]):
        variances = []
        for layer_states in liquid_states:
            if layer_states['combined'] is not None:
                var = layer_states['combined'].var().item()
                variances.append(var)
        
        if variances:
            current_stability = 1.0 / (1.0 + np.mean(variances))
            self.stability_history.append(current_stability)
            
            # Keep window size
            if len(self.stability_history) > self.stability_window:
                self.stability_history.pop(0)
            
            # Compute adaptive LR
            mean_stability = np.mean(self.stability_history)
            lr = self.base_lr * mean_stability
            
            # Update optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            return lr
        
        return self.base_lr


def curriculum_sequence_length(step: int, min_len: int = 64, max_len: int = 512, warmup_steps: int = 50000):
    progress = min(1.0, step / warmup_steps)
    return int(min_len + progress * (max_len - min_len))


# Example usage
# if __name__ == "__main__":
#     config = {
#         "vocab_size": 30522,
#         "embed_size": 768,
#         "num_heads": 12,
#         "num_experts": 4,
#         "num_layers": 6,
#         "max_seq_len": 512,
#         "dropout": 0.1
#     }
    
#     # Create model
#     model = LiquidTransformerLM(**config)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
    
#     # Example forward pass
#     batch_size = 2
#     seq_len = 128
#     input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
#     logits, liquid_states, aux_losses = model(input_ids)
#     print(f"Output shape: {logits.shape}")
#     print(f"Number of liquid state layers: {len(liquid_states)}")
#     print(f"Auxiliary losses: {list(aux_losses.keys())}")
    
#     # Example generation
#     prompt = torch.tensor([[101, 2023, 2003, 1037]])
#     generated = model.generate(prompt, max_length=50, temperature=0.8)
#     print(f"Generated shape: {generated.shape}")
