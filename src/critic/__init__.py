"""Critic module for translation efficiency and multi-metric prediction."""

from .critic_model import TranslationEfficiencyCritic, CriticTrainer
from .dataset import (
    TranslationEfficiencyDataset,
    load_zheng_data,
    create_data_loaders,
)
from .multi_metric_critic import MultiMetricCritic, MultiMetricTrainer
from .single_metric_critic import (
    SingleMetricCritic,
    SingleMetricTrainer,
    create_single_metric_loaders
)
from .critic_ensemble import CriticEnsemble

__all__ = [
    # Legacy single-task critic
    'TranslationEfficiencyCritic',
    'CriticTrainer',
    'TranslationEfficiencyDataset',
    'load_zheng_data',
    'create_data_loaders',
    # Multi-metric critic (single model, multiple heads)
    'MultiMetricCritic',
    'MultiMetricTrainer',
    # Single-metric critic (for separate datasets)
    'SingleMetricCritic',
    'SingleMetricTrainer',
    'create_single_metric_loaders',
    # Ensemble (combines multiple critics)
    'CriticEnsemble',
]

