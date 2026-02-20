# Improved AF-FCL: Modular Research Improvements
# 
# Each feature can be independently enabled via command-line flags:
#   --use_density_ratio       Feature 1: Density ratio credibility (replaces Eq. 8)
#   --use_personalized_nf     Feature 2: Personalized NF with KL regularization
#   --use_ema_extractor       Feature 3: EMA feature extractor for stable NF training
#   --use_fisher_aggregation  Feature 4: Fisher-weighted FedAvg for NF
#   --use_adaptive_theta      Feature 5: Task-similarity adaptive explore theta
#   --use_sinkhorn_kd         Feature 6: Sinkhorn divergence feature distillation (replaces Eq. 6)
#   --use_all_improvements    Enable all 6 features
#

FEATURE_FLAGS = [
    'use_density_ratio',
    'use_personalized_nf',
    'use_ema_extractor',
    'use_fisher_aggregation',
    'use_adaptive_theta',
    'use_sinkhorn_kd',
]

def any_improvement_enabled(args):
    """Check if any improvement feature is enabled."""
    return any(getattr(args, flag, False) for flag in FEATURE_FLAGS)
