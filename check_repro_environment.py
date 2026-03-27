from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CORE_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "jupyter": "jupyter",
    "notebook": "notebook",
    "ipykernel": "ipykernel",
}


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


def check_packages() -> list[str]:
    missing = [package for package, module in CORE_PACKAGES.items() if not _has_module(module)]
    _print_header("Python Packages")
    if missing:
        print("Missing:", ", ".join(missing))
    else:
        print("All core notebook/script dependencies are available.")
    return missing


def check_worldvalue_inputs() -> list[str]:
    required = [
        ROOT / "data" / "worldvalue" / "retained_questions_235.json",
        ROOT / "data" / "worldvalue" / "choices_to_numeric.json",
        ROOT / "data" / "worldvalue" / "population_response_clean.pkl",
        ROOT / "data" / "worldvalue" / "synthetic answers" / "clean" / "uniform_benchmark.pkl",
        ROOT / "data" / "worldvaluesbench" / "dataset_construction" / "question_metadata.json",
        ROOT / "data" / "worldvaluesbench" / "dataset_construction" / "codebook.json",
        ROOT / "data" / "worldvaluesbench" / "dataset_construction" / "answer_adjustment.json",
        ROOT / "data" / "worldvaluesbench" / "F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    _print_header("WorldValue Data")
    if missing:
        print("Missing extracted inputs:")
        for item in missing:
            print(f"  - {item}")
        print("Restore with:")
        print("  python datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout minimal")
    else:
        print("WorldValue minimal reproduction inputs are present.")
    return missing


def check_eedi_inputs() -> list[str]:
    required = [
        ROOT / "data" / "eedi" / "surveys.pkl",
        ROOT / "data" / "eedi" / "reports_interval_all.pkl",
        ROOT / "data" / "eedi" / "reports_point_all.pkl",
        ROOT / "data" / "eedi" / "synthetic_profiles.pkl",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    _print_header("EEDI Data")
    if missing:
        print("Missing extracted inputs:")
        for item in missing:
            print(f"  - {item}")
        print("Restore with:")
        print("  python datasets/unpack_reproduction_data.py --dataset eedi")
    else:
        print("EEDI reproduction inputs are present.")
    return missing


def check_opinionqa_inputs() -> list[str]:
    required = [
        ROOT / "data" / "opinionqa" / "surveys.pkl",
        ROOT / "data" / "opinionqa" / "reports_interval_all.pkl",
        ROOT / "data" / "opinionqa" / "reports_point_all.pkl",
        ROOT / "data" / "opinionqa" / "synthetic_profiles.pkl",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    _print_header("OpinionQA Data")
    if missing:
        print("Missing extracted inputs:")
        for item in missing:
            print(f"  - {item}")
        print("Restore with:")
        print("  python datasets/unpack_reproduction_data.py --dataset opinionqa")
    else:
        print("OpinionQA reproduction inputs are present.")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight check for running the paper reproduction on a fresh VM, Colab, or local machine."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "worldvalue", "eedi", "opinionqa"],
        default="all",
        help="Which reproduction workflow to validate.",
    )
    args = parser.parse_args()

    print(f"Repository root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print("This script only validates the environment and prints next steps; it does not install packages or modify files.")

    failures = []
    if check_packages():
        failures.append("packages")

    if args.dataset in {"all", "worldvalue"} and check_worldvalue_inputs():
        failures.append("worldvalue")
    if args.dataset in {"all", "eedi"} and check_eedi_inputs():
        failures.append("eedi")
    if args.dataset in {"all", "opinionqa"} and check_opinionqa_inputs():
        failures.append("opinionqa")

    _print_header("Next Step")
    if failures:
        print("Environment is not fully ready yet.")
        print("Install core dependencies with:")
        print("  pip install -r requirements.txt")
        print("Optional provider SDKs for fresh LLM generation only:")
        print("  pip install -r requirements-optional-llm.txt")
        return 1

    print("Environment looks ready for the selected reproduction workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
