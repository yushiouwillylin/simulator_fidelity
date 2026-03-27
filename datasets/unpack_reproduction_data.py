from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPRO_ROOT / "data"
WORLDVALUEBENCH_PREFIX = "World_Value/WorldValuesBench/"
WORLDVALUEBENCH_MINIMAL_FILES = {
    f"{WORLDVALUEBENCH_PREFIX}F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip":
        "F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip",
    f"{WORLDVALUEBENCH_PREFIX}dataset_construction/question_metadata.json":
        "dataset_construction/question_metadata.json",
    f"{WORLDVALUEBENCH_PREFIX}dataset_construction/codebook.json":
        "dataset_construction/codebook.json",
    f"{WORLDVALUEBENCH_PREFIX}dataset_construction/answer_adjustment.json":
        "dataset_construction/answer_adjustment.json",
}


def _extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    if member.endswith("/"):
        dest.mkdir(parents=True, exist_ok=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def _extract_with_prefix(zf: zipfile.ZipFile, prefix: str, dest_root: Path) -> int:
    count = 0
    for member in zf.namelist():
        if not member.startswith(prefix):
            continue
        rel = Path(member).relative_to(prefix)
        if str(rel) == ".":
            continue
        _extract_member(zf, member, dest_root / rel)
        count += 1
    return count


def _extract_selected_files(zf: zipfile.ZipFile, members_to_dest: dict[str, Path]) -> int:
    count = 0
    available = set(zf.namelist())
    for member, dest in members_to_dest.items():
        if member not in available:
            continue
        _extract_member(zf, member, dest)
        count += 1
    return count


def _extract_worldvaluesbench_minimal(zf: zipfile.ZipFile, raw_root: Path) -> int:
    file_targets = {
        member: raw_root / relative_dest
        for member, relative_dest in WORLDVALUEBENCH_MINIMAL_FILES.items()
    }
    return _extract_selected_files(zf, file_targets)


def unpack_worldvalue(force: bool = False, layout: str = "minimal") -> None:
    archive = REPRO_ROOT / "datasets" / "worldvalue_data.zip"
    processed_root = DATA_ROOT / "worldvalue"
    raw_root = DATA_ROOT / "worldvaluesbench"
    if force:
        shutil.rmtree(processed_root, ignore_errors=True)
        shutil.rmtree(raw_root, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        n1 = _extract_with_prefix(zf, "Data/WorldValue/", processed_root)
        if layout == "full":
            n2 = _extract_with_prefix(zf, WORLDVALUEBENCH_PREFIX, raw_root)
        else:
            n2 = _extract_worldvaluesbench_minimal(zf, raw_root)
    print(f"WorldValue extracted: {n1} processed entries -> {processed_root}")
    print(f"WorldValuesBench extracted ({layout}): {n2} entries -> {raw_root}")


def unpack_eedi(force: bool = False) -> None:
    archive = REPRO_ROOT / "datasets" / "eedi_data.zip"
    dest_root = DATA_ROOT / "eedi"
    if force:
        shutil.rmtree(dest_root, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        n = _extract_with_prefix(zf, "Data/EEDI/", dest_root)
    print(f"EEDI extracted: {n} entries -> {dest_root}")


def unpack_opinionqa(force: bool = False) -> None:
    archive = REPRO_ROOT / "datasets" / "opinionqa_data.zip"
    dest_root = DATA_ROOT / "opinionqa"
    if force:
        shutil.rmtree(dest_root, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        n = _extract_with_prefix(zf, "Data/OpinionQA/", dest_root)
    print(f"OpinionQA extracted: {n} entries -> {dest_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack reproduction archives into the bundle-local data/ tree.")
    parser.add_argument(
        "--dataset",
        choices=["all", "worldvalue", "eedi", "opinionqa"],
        default="all",
        help="Which dataset bundle to unpack.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing extracted targets before unpacking.",
    )
    parser.add_argument(
        "--worldvalue-layout",
        choices=["minimal", "full"],
        default="minimal",
        help=(
            "How much of the WorldValuesBench tree to extract. "
            "'minimal' keeps only the pieces needed for the paper reproduction; "
            "'full' restores the full upstream bundle."
        ),
    )
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"all", "worldvalue"}:
        unpack_worldvalue(force=args.force, layout=args.worldvalue_layout)
    if args.dataset in {"all", "eedi"}:
        unpack_eedi(force=args.force)
    if args.dataset in {"all", "opinionqa"}:
        unpack_opinionqa(force=args.force)


if __name__ == "__main__":
    main()
