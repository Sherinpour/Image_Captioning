import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


INVISIBLE_CHARS = [
    "\u200c",  # zero-width non-joiner
    "\u200f",  # right-to-left mark
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
    "\xa0",    # non-breaking space
]

ARABIC_TO_PERSIAN_MAP = {
    "ي": "ی",
    "ك": "ک",
}


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if not isinstance(value, str):
        value = str(value)
    # Unicode normalization
    value = unicodedata.normalize("NFC", value)
    # Remove invisible characters
    for ch in INVISIBLE_CHARS:
        value = value.replace(ch, " ")
    # Collapse whitespace
    value = " ".join(value.split())
    # Map Arabic forms to Persian
    for src, dst in ARABIC_TO_PERSIAN_MAP.items():
        value = value.replace(src, dst)
    # Final strip
    return value.strip()


def ensure_entities_list_of_dicts(entities_val: Any) -> List[Dict[str, Any]]:
    if entities_val is None or (isinstance(entities_val, float) and pd.isna(entities_val)):
        return []
    if isinstance(entities_val, list):
        # Keep only dict items
        return [e for e in entities_val if isinstance(e, dict)]
    if isinstance(entities_val, dict):
        return [entities_val]
    # Unsupported types → empty list
    return []


def join_values(values: Any) -> Optional[str]:
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return None
    if isinstance(values, list):
        # Flatten nested lists of primitives/strings
        flat: List[str] = []
        for v in values:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if isinstance(v, list):
                flat.extend([str(x) for x in v if x is not None and not (isinstance(x, float) and pd.isna(x))])
            else:
                flat.append(str(v))
        return "; ".join(dict.fromkeys([s.strip() for s in flat if s.strip()])) or None
    # Primitive
    s = str(values).strip()
    return s or None


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize string columns
    for col in ["title", "group", "product"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_text)

    # Deduplicate by random_key if present
    if "random_key" in df.columns:
        df = df.drop_duplicates(subset=["random_key"], keep="first").reset_index(drop=True)

    # Ensure entities is a list[dict]
    if "entities" in df.columns:
        df["entities"] = df["entities"].apply(ensure_entities_list_of_dicts)
    else:
        df["entities"] = [[] for _ in range(len(df))]

    # Fill empties: For text fields use "نامشخص"; leave ids/keys as is
    for col in ["title", "group", "product"]:
        if col in df.columns:
            df[col] = df[col].fillna("نامشخص").replace("", "نامشخص")

    return df


def explode_entities(df: pd.DataFrame) -> pd.DataFrame:
    # Explode entities into rows
    exploded = df.explode("entities", ignore_index=True)
    # If entities is empty, rows will have NaN; keep original row with NaN entity
    # Extract fields
    exploded["entity_name"] = exploded["entities"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    exploded["entity_values_raw"] = exploded["entities"].apply(lambda x: x.get("values") if isinstance(x, dict) else None)
    exploded["entity_values"] = exploded["entity_values_raw"].apply(join_values)
    # Clean entity_name
    exploded["entity_name"] = exploded["entity_name"].map(normalize_text)
    return exploded


def pivot_entities_wide(exploded: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows with a valid entity_name
    valid = exploded.dropna(subset=["entity_name"]).copy()
    # If multiple rows per key+entity_name, aggregate values by dedup-join
    index_cols = [c for c in ["random_key"] if c in valid.columns]
    # Use random_key as index if present, otherwise fallback to index
    if index_cols:
        grp = valid.groupby(index_cols + ["entity_name"], dropna=False)["entity_values"].apply(
            lambda s: "; ".join(dict.fromkeys([v for v in s.dropna().astype(str) if v.strip()])) or None
        )
        wide = grp.unstack("entity_name")
        wide = wide.reset_index()
    else:
        # Fallback: include row index to avoid collapsing unrelated rows
        valid = valid.reset_index().rename(columns={"index": "_row"})
        grp = valid.groupby(["_row", "entity_name"], dropna=False)["entity_values"].apply(
            lambda s: "; ".join(dict.fromkeys([v for v in s.dropna().astype(str) if v.strip()])) or None
        )
        wide = grp.unstack("entity_name").reset_index().drop(columns=["_row"])  # type: ignore[assignment]

    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and normalize entities DataFrame")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "main_entities_dataframe.parquet",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "main_entities_dataframe.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()

    # Prefer parquet if exists
    if args.input_parquet.exists():
        df = pd.read_parquet(args.input_parquet)
    elif args.input_csv.exists():
        df = pd.read_csv(args.input_csv)
    else:
        raise FileNotFoundError("No input file found (parquet or csv)")

    df_clean = preprocess_dataframe(df.copy())
    exploded = explode_entities(df_clean.copy())
    wide = pivot_entities_wide(exploded)

    # Merge wide back with identifier columns for convenience
    id_cols = [c for c in ["random_key", "title", "group", "product", "image_url"] if c in df_clean.columns]
    if id_cols:
        wide = df_clean[id_cols].drop_duplicates(subset=["random_key"] if "random_key" in id_cols else None).merge(
            wide, on="random_key", how="left"
        ) if "random_key" in id_cols else wide

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save cleaned
    cleaned_parquet = args.output_dir / "main_entities_dataframe.cleaned.parquet"
    cleaned_csv = args.output_dir / "main_entities_dataframe.cleaned.csv"
    df_clean.to_parquet(cleaned_parquet, index=False)
    df_clean.to_csv(cleaned_csv, index=False)

    # Save exploded
    exploded_parquet = args.output_dir / "main_entities_dataframe.exploded.parquet"
    exploded_csv = args.output_dir / "main_entities_dataframe.exploded.csv"
    exploded.to_parquet(exploded_parquet, index=False)
    exploded.to_csv(exploded_csv, index=False)

    # Save wide
    wide_parquet = args.output_dir / "main_entities_dataframe.wide.parquet"
    wide_csv = args.output_dir / "main_entities_dataframe.wide.csv"
    wide.to_parquet(wide_parquet, index=False)
    wide.to_csv(wide_csv, index=False)

    print({
        "cleaned": str(cleaned_parquet),
        "exploded": str(exploded_parquet),
        "wide": str(wide_parquet),
        "rows_cleaned": len(df_clean),
        "rows_exploded": len(exploded),
        "rows_wide": len(wide),
    })


if __name__ == "__main__":
    main()


