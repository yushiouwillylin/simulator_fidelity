import json
import re
import sys
from pathlib import Path

import numpy as np


def is_q7_to_q17(qid: str) -> bool:
    s = str(qid).strip()
    m = re.match(r"(?i)^q\s*0*(\d+)$", s) or re.match(r"^0*(\d+)$", s)
    return bool(m and 7 <= int(m.group(1)) <= 17)


def load_retained_questions(
    json_path: str = "data/worldvalue/retained_questions_235.json",
    fallback_npy: str = "data/worldvalue/good_questions.npy",
) -> list[str]:
    json_file = Path(json_path)
    if json_file.exists():
        return [str(q) for q in json.loads(json_file.read_text())]

    questions = np.load(fallback_npy, allow_pickle=True).tolist()
    return [str(q) for q in questions if not is_q7_to_q17(q)]


def filter_mapping_to_questions(mapping, questions: list[str]):
    keep = {str(q) for q in questions}
    return {str(k): v for k, v in mapping.items() if str(k) in keep}


def install_numpy_pickle_compat() -> None:
    """Alias older NumPy pickle module paths used by some committed artifacts."""
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


def find_repo_root(start: Path) -> Path:
    start = Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "datasets" / "worldvalue_data.zip").exists() and candidate.name == "paper_reproduction":
            return candidate
        if (candidate / "paper_reproduction" / "datasets" / "worldvalue_data.zip").exists():
            return candidate / "paper_reproduction"
        if (candidate / "worldvalue_quantile" / "wvs_notebook_helpers.py").exists() and (
            candidate / "datasets" / "worldvalue_data.zip"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the paper_reproduction root containing datasets/worldvalue_data.zip."
    )


def worldvalue_required_inputs(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    return {
        "retained_questions": root / "data" / "worldvalue" / "retained_questions_235.json",
        "good_questions": root / "data" / "worldvalue" / "good_questions.npy",
        "choices_to_numeric": root / "data" / "worldvalue" / "choices_to_numeric.json",
        "question_metadata": root / "data" / "worldvaluesbench" / "dataset_construction" / "question_metadata.json",
        "codebook": root / "data" / "worldvaluesbench" / "dataset_construction" / "codebook.json",
        "answer_adjustment": root / "data" / "worldvaluesbench" / "dataset_construction" / "answer_adjustment.json",
        "wvs_raw_zip": root / "data" / "worldvaluesbench" / "F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip",
        "worldvalue_zip": root / "datasets" / "worldvalue_data.zip",
    }


def worldvalue_restore_instructions(root: Path) -> str:
    root = Path(root).resolve()
    return (
        "Restore the internal WorldValue bundle data with:\n"
        "  python datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout minimal"
    )


def ensure_worldvalue_inputs(root: Path) -> dict[str, Path]:
    required = worldvalue_required_inputs(root)
    missing = [
        str(path.relative_to(root))
        for name, path in required.items()
        if name != "worldvalue_zip" and not path.exists()
    ]
    if missing:
        missing_list = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Missing required WorldValue inputs:\n"
            f"{missing_list}\n"
            f"{worldvalue_restore_instructions(root)}"
        )
    return required


def worldvalue_figures_dir(root: Path) -> Path:
    path = Path(root).resolve() / "worldvalue_quantile" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def worldvalue_clean_outputs_dir(root: Path) -> Path:
    path = Path(root).resolve() / "data" / "worldvalue" / "synthetic answers" / "clean"
    path.mkdir(parents=True, exist_ok=True)
    return path
