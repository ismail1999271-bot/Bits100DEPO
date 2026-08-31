"""Run a deterministic demo of the scoring pipeline.

Usage:
    python -m bist_hunter.demo
"""

from .market_regime import RegimeFeatures, fear_greed_score, regime_label
from .scoring import CandidateFeatures, score_candidate


def main() -> None:
    regime = RegimeFeatures(
        momentum=0.72,
        breadth=0.68,
        volume_risk_appetite=0.74,
        volatility_inverse=0.58,
        global_risk_appetite=0.61,
        flow=0.77,
    )
    market_score = fear_greed_score(regime)
    print(f"BIST Fear/Greed baseline: {market_score:.2f}/100 ({regime_label(market_score)})")

    candidates = {
        "DEMO_A": CandidateFeatures(.95, .82, .88, .90, .76, .84, .70, .08),
        "DEMO_B": CandidateFeatures(.72, .75, .61, .68, .70, .55, .82, .25),
        "DEMO_C": CandidateFeatures(.58, .64, .55, .40, .62, .42, .45, .65),
    }
    print("\nTAVAN RADAR")
    for symbol, features in sorted(candidates.items(), key=lambda item: score_candidate(item[1])[0], reverse=True):
        score, reasons = score_candidate(features)
        print(f"{symbol:8} {score:5.1f}/100  drivers={', '.join(reasons) or 'none'}")


if __name__ == "__main__":
    main()
