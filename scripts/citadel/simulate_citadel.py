from acn.citadel import CitadelActionRequest, CitadelSafetyLayer
from acn.controller import AdaptiveAction


def main() -> None:
    citadel = CitadelSafetyLayer()
    result = citadel.validate_action(
        CitadelActionRequest(
            action=AdaptiveAction.INCREASE_LEARNING_RATE,
            actor="operator",
            branch_name="main",
            parameters={"learning_rate": 2.0},
        )
    )

    print(f"allowed={result.allowed}")
    print(f"decision={result.decision.value}")
    for reason in result.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
