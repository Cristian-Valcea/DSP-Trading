"""
Double DQN with Dueling Architecture for Multi-Symbol Trading

Architecture:
- Per-symbol CNN+LSTM encoder for temporal features
- Cross-symbol attention for market-wide context
- Portfolio state integration
- Dueling heads (Value + Advantage) per symbol
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SymbolEncoder(nn.Module):
    """
    Encode temporal features for a single symbol.

    Input: (batch, window=60, features=30)
    Output: (batch, hidden=128)
    """

    def __init__(
        self,
        input_features: int = 30,
        conv_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Temporal convolutions
        self.conv1 = nn.Conv1d(input_features, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels * 2)

        # LSTM for sequential patterns
        self.lstm = nn.LSTM(
            input_size=conv_channels * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window, features)

        Returns:
            hidden: (batch, hidden_size)
        """
        # Transpose for conv1d: (batch, features, window)
        x = x.transpose(1, 2)

        # Convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transpose back for LSTM: (batch, window, channels)
        x = x.transpose(1, 2)

        # LSTM - take last hidden state
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]  # (batch, hidden_size)

        return self.dropout(hidden)


class CrossSymbolAttention(nn.Module):
    """
    Multi-head attention across symbols for market-wide context.

    Input: (batch, num_symbols, hidden)
    Output: (batch, num_symbols, hidden)
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_symbols, hidden)

        Returns:
            out: (batch, num_symbols, hidden)
        """
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + self.dropout(attn_out))
        return out


class DuelingHead(nn.Module):
    """
    Dueling network head for a single symbol.

    Computes Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    Input: (batch, hidden)
    Output: (batch, num_actions)
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_actions: int = 5,
    ):
        super().__init__()

        # Value stream
        self.value_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream
        self.advantage_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden)

        Returns:
            q_values: (batch, num_actions)
        """
        value = self.value_fc(x)  # (batch, 1)
        advantage = self.advantage_fc(x)  # (batch, num_actions)

        # Dueling combination
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class DuelingDQN(nn.Module):
    """
    Complete Dueling Double DQN for multi-symbol trading.

    Observation Space (from Gate 1):
        - rolling_window: (batch, window=60, num_symbols=9, features=30)
        - portfolio_state: (batch, 21)

    Action Space:
        - (batch, num_symbols=9, num_actions=5)
    """

    def __init__(
        self,
        num_symbols: int = 9,
        window_size: int = 60,
        num_features: int = 30,
        portfolio_size: int = 21,
        num_actions: int = 5,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_symbols = num_symbols
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # Per-symbol encoders (shared weights)
        self.symbol_encoder = SymbolEncoder(
            input_features=num_features,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # Cross-symbol attention
        self.cross_attention = CrossSymbolAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Portfolio integration
        self.portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combine market encoding with portfolio
        self.integration_fc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-symbol dueling heads (shared weights)
        self.dueling_head = DuelingHead(
            hidden_size=hidden_size,
            num_actions=num_actions,
        )

    def forward(
        self,
        rolling_window: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute Q-values for all symbols.

        Args:
            rolling_window: (batch, window, num_symbols, features)
            portfolio_state: (batch, portfolio_size)

        Returns:
            q_values: (batch, num_symbols, num_actions)
        """
        batch_size = rolling_window.shape[0]

        # Encode each symbol's features
        # Reshape: (batch, window, symbols, features) -> (batch * symbols, window, features)
        rolling_window = rolling_window.permute(0, 2, 1, 3)  # (batch, symbols, window, features)
        rolling_window = rolling_window.reshape(-1, rolling_window.shape[2], rolling_window.shape[3])

        symbol_encodings = self.symbol_encoder(rolling_window)  # (batch * symbols, hidden)
        symbol_encodings = symbol_encodings.reshape(batch_size, self.num_symbols, -1)  # (batch, symbols, hidden)

        # Cross-symbol attention
        symbol_encodings = self.cross_attention(symbol_encodings)  # (batch, symbols, hidden)

        # Encode portfolio state
        portfolio_encoding = self.portfolio_fc(portfolio_state)  # (batch, hidden // 2)

        # Compute Q-values for each symbol
        q_values_list = []
        for i in range(self.num_symbols):
            # Get symbol encoding
            symbol_hidden = symbol_encodings[:, i, :]  # (batch, hidden)

            # Integrate with portfolio
            combined = torch.cat([symbol_hidden, portfolio_encoding], dim=1)
            integrated = self.integration_fc(combined)  # (batch, hidden)

            # Compute Q-values
            q_vals = self.dueling_head(integrated)  # (batch, num_actions)
            q_values_list.append(q_vals)

        # Stack Q-values: (batch, num_symbols, num_actions)
        q_values = torch.stack(q_values_list, dim=1)

        return q_values

    def select_actions(
        self,
        rolling_window: torch.Tensor,
        portfolio_state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions using epsilon-greedy policy.

        Args:
            rolling_window: (batch, window, num_symbols, features)
            portfolio_state: (batch, portfolio_size)
            epsilon: Exploration rate

        Returns:
            actions: (batch, num_symbols) selected actions
            q_values: (batch, num_symbols, num_actions) Q-values
        """
        with torch.no_grad():
            q_values = self.forward(rolling_window, portfolio_state)

        batch_size = q_values.shape[0]
        actions = np.zeros((batch_size, self.num_symbols), dtype=np.int32)

        for b in range(batch_size):
            if np.random.random() < epsilon:
                # Random exploration
                actions[b] = np.random.randint(0, self.num_actions, size=self.num_symbols)
            else:
                # Greedy exploitation
                actions[b] = q_values[b].argmax(dim=1).cpu().numpy()

        return actions, q_values.cpu().numpy()


def create_model(
    num_symbols: int = 9,
    window_size: int = 60,
    num_features: int = 30,
    portfolio_size: int = 21,
    num_actions: int = 5,
    hidden_size: int = 128,
    device: str = "cpu",
) -> DuelingDQN:
    """
    Factory function to create DQN model.

    Args:
        num_symbols: Number of symbols to trade
        window_size: Rolling window size
        num_features: Features per symbol per timestep
        portfolio_size: Portfolio state dimensions
        num_actions: Actions per symbol
        hidden_size: Hidden layer size
        device: Device to place model on

    Returns:
        DuelingDQN model
    """
    model = DuelingDQN(
        num_symbols=num_symbols,
        window_size=window_size,
        num_features=num_features,
        portfolio_size=portfolio_size,
        num_actions=num_actions,
        hidden_size=hidden_size,
    )

    return model.to(device)


# Test code
if __name__ == "__main__":
    # Test model construction and forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing DuelingDQN on {device}")

    model = create_model(device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    rolling_window = torch.randn(batch_size, 60, 9, 30).to(device)
    portfolio_state = torch.randn(batch_size, 21).to(device)

    q_values = model(rolling_window, portfolio_state)
    print(f"Q-values shape: {q_values.shape}")  # Should be (4, 9, 5)

    # Test action selection
    actions, q_vals = model.select_actions(rolling_window, portfolio_state, epsilon=0.1)
    print(f"Actions shape: {actions.shape}")  # Should be (4, 9)

    print("✅ Model test passed!")
