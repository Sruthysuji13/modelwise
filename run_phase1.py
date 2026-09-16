"""End-to-end smoke test for Phase 1, mirroring the intended UI flow in text form."""
import sys
from services.data_loader import load_dataset, DatasetLoadError
from services.data_profiler import (
    profile_dataset, compute_data_quality_score, suggest_target_column, detect_problem_type,
)
from services.data_cleaner import clean_dataset
from services.leakage_detector import detect_leakage
from services.imbalance_handler import detect_class_imbalance
from services.model_trainer import TrainingConfig, train_models
from services.evaluator import build_leaderboard
from utils.safety_checks import run_safety_checks


def main(file_path: str, target: str | None = None, tune: bool = False):
    try:
        df, file_info = load_dataset(file_path)
    except DatasetLoadError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print("File info:", file_info)

    profile = profile_dataset(df)
    quality = compute_data_quality_score(df, profile)
    print(f"\nData Quality Score: {quality['score']}/100")
    for issue in quality["issues"]:
        print(f"  • {issue}")

    if target is None:
        suggestion = suggest_target_column(df)
        target = suggestion["suggested_target"]
        print(f"\nSuggested target: {target} (confidence: {suggestion['confidence']})")

    problem = detect_problem_type(df, target)
    print(f"\nDetected problem type: {problem['problem_type']}")
    print(f"Reason: {problem['reason']}")

    safety = run_safety_checks(df, target, problem["problem_type"])
    for w in safety["warnings"]:
        print(f"WARNING: {w}")
    if safety["blocking"]:
        print("Training cannot proceed — see warnings above.")
        sys.exit(1)

    df = df.dropna(subset=[target])

    leakage = detect_leakage(df, target, problem["problem_type"])
    if leakage:
        print("\nPotential data leakage:")
        for item in leakage:
            print(f"  Column: {item['column']} — {item['reason']}")
        leak_cols = [item["column"] for item in leakage]
        df = df.drop(columns=leak_cols)
        print(f"  → Auto-excluded from training: {leak_cols}")

    if problem["problem_type"] == "classification":
        imbalance = detect_class_imbalance(df, target)
        if imbalance.get("imbalanced"):
            print("\nClass imbalance detected:", imbalance["distribution"])
            print("Recommendation:", imbalance["recommendation"])

    df, clean_summary = clean_dataset(df, target)
    print("\nCleaning Summary:")
    for a in clean_summary["actions"]:
        print(f"  ✓ {a}")

    config = TrainingConfig(tune_hyperparameters=tune)
    print("\nTraining models...")
    output = train_models(df, target, problem["problem_type"], config)

    for r in output["results"]:
        if r["status"] == "success":
            print(f"  ✓ {r['model_name']} — CV mean: {r['cv']['mean']} (± {r['cv']['std']})")
            if r.get("best_hyperparameters"):
                after = r.get("cv_after_tuning", {}).get("mean", "n/a")
                print(f"      Tuned — before: {r['cv']['mean']} → after: {after}")
                print(f"      Best hyperparameters: {r['best_hyperparameters']}")
        elif r["status"] == "skipped":
            print(f"  ⊘ {r['model_name']} — skipped: {r['error']}")
        else:
            print(f"  ✗ {r['model_name']} — failed: {r['error']}")

    leaderboard = build_leaderboard(output["results"], problem["problem_type"])
    print("\nLeaderboard:")
    for r in leaderboard:
        print(f"  {r['rank']}. {r['model_name']} — {r['metrics']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    tune = "--tune" in args
    positional = [a for a in args if not a.startswith("--")]
    path = positional[0] if len(positional) > 0 else "sample.csv"
    tgt = positional[1] if len(positional) > 1 else None
    main(path, tgt, tune)