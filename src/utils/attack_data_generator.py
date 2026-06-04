"""
attack_data_generator.py
========================

Attack data generation class with 6 Byzantine attack types:
S1: Linear Drift
S2: Exponential Surge
S3: Wave Oscillation
S4: FDI Step Change
S5: Persistent Bias
S6: All Nodes Compromised
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "data"))

from gat_data_generator import SyntheticDataGenerator


class AttackDataGenerator:
    """Generate synthetic data with 6 Byzantine attack types."""

    def __init__(self, num_nodes: int = 5, seq_len: int = 100, num_samples: int = 100):
        """Initialize attack generator.
        
        Args:
            num_nodes: Number of sensor nodes
            seq_len: Sequence length
            num_samples: Number of samples to generate
        """
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.gen = SyntheticDataGenerator(
            num_nodes=num_nodes,
            sequence_length=seq_len,
            num_samples_per_class=num_samples,
            random_seed=42,
        )

    def _base_natural(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base normal data without attacks."""
        X, y, a = self.gen.generate_dataset()
        idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
        return X[idx], a[idx]

    def _make_dataset(
        self,
        X_natural: np.ndarray,
        X_attacked: np.ndarray,
        attrs_attacked: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine normal and attacked data into final dataset."""
        X = np.concatenate([X_natural, X_attacked], axis=0)
        y = np.concatenate(
            [np.zeros(len(X_natural)), np.ones(len(X_attacked))], axis=0
        )
        attrs = np.concatenate(
            [np.zeros_like(X_natural[:, :, 0]), attrs_attacked], axis=0
        )
        return X, y, attrs

    # ========================================================================
    # S1: Linear Drift Attack
    # ========================================================================
    def linear_drift(self, delta: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S1: Linear drift attack where selected sensors linearly drift from normal."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 3)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        for s in attacked_indices:
            for t in range(self.seq_len):
                drift = delta * (t + 1)
                X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S2: Exponential Surge Attack
    # ========================================================================
    def exponential_drift(
        self, delta: float = 0.02, alpha: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S2: Rapid exponential surge attack - sensors rapidly spike mid-sequence."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 2)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        onset_t = self.seq_len // 3
        for s in attacked_indices:
            for t in range(self.seq_len):
                if t >= onset_t:
                    exp_term = np.exp(alpha * (t - onset_t) / (self.seq_len - onset_t))
                    drift = delta * 5.0 * (exp_term - 1.0)
                    X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S3: Wave Oscillation Attack
    # ========================================================================
    def frogging_attack(
        self, delta: float = 0.02, switch_period: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S3: Regular oscillation attack - sensors oscillate with large amplitude."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 2)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        for s in attacked_indices:
            for t in range(self.seq_len):
                cycle = (t // switch_period) % 2
                drift = (delta * 10.0) if cycle == 0 else -(delta * 10.0)
                X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S4: FDI Step Change Attack
    # ========================================================================
    def fdi_step_change(
        self, magnitude: float = 2.0, onset_frac: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S4: FDI step change attack where values suddenly change at a point."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 3)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        onset_t = int(onset_frac * self.seq_len)
        for s in attacked_indices:
            X_attacked[:, s, onset_t:] += magnitude

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S5: Persistent Bias Attack
    # ========================================================================
    def polynomial_drift(
        self, delta: float = 0.015, power: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S5: Persistent bias attack - steady offset added to sensors."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 2)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        bias_magnitude = 0.5
        for s in attacked_indices:
            X_attacked[:, s, :] += bias_magnitude

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S6: All Nodes Compromised
    # ========================================================================
    def majority_compromised(
        self, delta: float = 0.002
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S6: ALL nodes compromised with subtle drift - fundamental limitation."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        attacked_indices = np.arange(self.num_nodes)

        for s in attacked_indices:
            for t in range(self.seq_len):
                drift = delta * (t + 1) / self.seq_len
                X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.ones((self.num_samples, self.num_nodes), dtype=np.float32)

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)
