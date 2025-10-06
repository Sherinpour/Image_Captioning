import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "title",
    "group",
    "product",
    "entities",
    "image_url",
    "random_key",
]


def read_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Read a JSON file that may contain a list of records or a single record.

    Returns a list of dict rows.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # If the dict wraps the records under a common key, try flattening common patterns
        for key in ("data", "items", "records", "rows"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {file_path}")


def iter_all_records(files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for fp in files:
        try:
            for rec in read_json_file(fp):
                # Attach source filename for traceability
                rec.setdefault("_source_file", fp.name)
                yield rec
        except Exception as exc:
            # Fail fast but with context
            raise RuntimeError(f"Failed to parse {fp}: {exc}") from exc


def normalize_records_to_dataframe(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(list(records))
    # Ensure required columns exist; if missing, create with NaN
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Reorder to put required columns first
    ordered_cols = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[ordered_cols]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after normalization: {missing_cols}")

    # Optionally check for rows with nulls in required fields
    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    # Not an error, but print a summary to stderr for awareness
    sys.stderr.write(f"Nulls in required columns:\n{null_counts.to_dict()}\n")


def build_dataframe(input_dir: Path, pattern: str = "*_entities_dataset_v2.json") -> pd.DataFrame:
    files = sorted(input_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir}")
    df = normalize_records_to_dataframe(iter_all_records(files))
    validate_required_columns(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main entities DataFrame from JSON shards")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "entities_dataset_v2",
        help="Directory containing *_entities_dataset_v2.json files",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "main_entities_dataframe.csv",
        help="Path to write combined CSV",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "main_entities_dataframe.parquet",
        help="Path to write combined Parquet",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_entities_dataset_v2.json",
        help="Glob pattern for input JSON files (recursive)",
    )
    args = parser.parse_args()

    df = build_dataframe(args.input_dir, args.pattern)

    # Save outputs
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    try:
        df.to_parquet(args.output_parquet, index=False)
    except Exception as exc:
        sys.stderr.write(f"Warning: failed to write parquet: {exc}\n")

    print(
        {
            "rows": len(df),
            "cols": len(df.columns),
            "csv": str(args.output_csv),
            "parquet": str(args.output_parquet),
        }
    )


if __name__ == "__main__":
    main()


