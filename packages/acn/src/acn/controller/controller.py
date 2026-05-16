import logging
from collections.abc import Sequence

from acn.controller.domain import ControllerDecision, MetricPoint, TrainingContext
from acn.controller.policies import RuleBasedAdaptivePolicy

logger = logging.getLogger(__name__)


class AdaptiveController:
    def __init__(self, policy: RuleBasedAdaptivePolicy | None = None) -> None:
        self._policy = policy or RuleBasedAdaptivePolicy()

    def decide(
        self,
        *,
        metrics: Sequence[MetricPoint],
        context: TrainingContext,
    ) -> ControllerDecision:
        decision = self._policy.decide(metrics=metrics, context=context)
        logger.info(
            "controller.decision",
            extra={
                "action": decision.action.value,
                "confidence": decision.confidence,
                "branch": context.branch_name,
                "current_commit_id": context.current_commit_id,
                "reasons": list(decision.reasons),
            },
        )
        return decision
