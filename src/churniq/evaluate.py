import json
from pathlib import Path

from churniq.data import load_csv, validate_target
from churniq.train import save_best, train_models


def run(dataset: str, output_dir: str = "artifacts") -> dict:
    df = load_csv(dataset)
    validate_target(df)
    result = train_models(df)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = save_best(result, "models")
    report = {
        "best_model": result["best_model"],
        "metrics": result["results"],
        "model_path": str(model_path),
    }
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), indent=2))
