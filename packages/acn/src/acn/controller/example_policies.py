from acn.controller.domain import AdaptiveAction
from acn.controller.policies import RuleBasedPolicyConfig


def conservative_policy() -> RuleBasedPolicyConfig:
    return RuleBasedPolicyConfig(
        degradation_patience=2,
        degradation_min_delta=0.03,
        plateau_window=5,
        plateau_min_delta=0.003,
        plateau_action=AdaptiveAction.DECREASE_LEARNING_RATE,
        learning_rate_decrease_factor=0.5,
    )


def exploratory_policy() -> RuleBasedPolicyConfig:
    return RuleBasedPolicyConfig(
        degradation_patience=3,
        degradation_min_delta=0.05,
        plateau_window=4,
        plateau_min_delta=0.01,
        plateau_action=AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
        learning_rate_increase_factor=1.5,
    )
