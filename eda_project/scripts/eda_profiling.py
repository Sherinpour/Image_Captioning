import argparse
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def safe_read_parquet_or_csv(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    raise FileNotFoundError(f"No input found: {path_parquet} or {path_csv}")


def dataframe_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer, memory_usage="deep")
    return buffer.getvalue()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    pct = (counts / len(df)) * 100 if len(df) else 0
    return pd.DataFrame({"missing_count": counts, "missing_pct": pct}).sort_values("missing_count", ascending=False)


def make_profile_report(df: pd.DataFrame, output_html: Path) -> Optional[Path]:
    try:
        # Prefer ydata_profiling; fallback to pandas_profiling legacy name
        try:
            from ydata_profiling import ProfileReport  # type: ignore
        except Exception:  # pragma: no cover
            import pandas_profiling as pp  # type: ignore

            ProfileReport = pp.ProfileReport  # type: ignore

        profile = ProfileReport(df, title="Data Profile", minimal=True)
        profile.to_file(str(output_html))
        return output_html
    except Exception:
        return None


def build_report(cleaned: pd.DataFrame, exploded: Optional[pd.DataFrame]) -> str:
    lines = []
    lines.append(f"# EDA Profile\n\nGenerated: {datetime.utcnow().isoformat()}Z\n")

    # Basic info
    lines.append("## Cleaned DataFrame info\n")
    lines.append("```\n" + dataframe_info(cleaned) + "```\n")
    ms_clean = missing_summary(cleaned)
    lines.append("## Missing values (cleaned)\n")
    lines.append(ms_clean.to_markdown())
    lines.append("\n")
    lines.append(f"- Rows: {len(cleaned)}  | Cols: {len(cleaned.columns)}\n")
    # Duplicates: prefer random_key if present to avoid unhashable object columns
    if "random_key" in cleaned.columns:
        dup_count = int(cleaned["random_key"].duplicated().sum())
    else:
        # Fallback: stringify to handle lists/dicts safely (slower but robust)
        dup_count = int(cleaned.astype(str).duplicated().sum())
    lines.append(f"- Duplicates (by key/full-row-safe): {dup_count}\n")

    # Title length distribution
    if "title" in cleaned.columns:
        tl = cleaned["title"].astype(str).str.len().describe()
        lines.append("## Title length describe (cleaned)\n")
        lines.append("```\n" + str(tl) + "\n```)\n")

    # Uniques for group/product
    for col in ["group", "product"]:
        if col in cleaned.columns:
            lines.append(f"- Unique {col}: {int(cleaned[col].nunique(dropna=True))}\n")

    # Entities analysis from exploded
    if exploded is not None and not exploded.empty:
        lines.append("\n## Entities (from exploded)\n")
        name_col = "entity_name" if "entity_name" in exploded.columns else ("name" if "name" in exploded.columns else None)
        if name_col is not None:
            if "product" in exploded.columns:
                nunq = exploded.groupby("product")[name_col].nunique(dropna=True).sort_values(ascending=False).head(20)
                lines.append("Top 20 products by unique entity names:\n")
                lines.append("```\n" + str(nunq) + "\n```)\n")
            total_entities = int(exploded[name_col].nunique(dropna=True))
            lines.append(f"- Unique entity names: {total_entities}\n")
        else:
            lines.append("- No entity name column found in exploded data.\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA profiling for cleaned/exploded data")
    base_data = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument("--cleaned-parquet", type=Path, default=base_data / "main_entities_dataframe.cleaned.parquet")
    parser.add_argument("--cleaned-csv", type=Path, default=base_data / "main_entities_dataframe.cleaned.csv")
    parser.add_argument("--exploded-parquet", type=Path, default=base_data / "main_entities_dataframe.exploded.parquet")
    parser.add_argument("--exploded-csv", type=Path, default=base_data / "main_entities_dataframe.exploded.csv")
    parser.add_argument("--output-md", type=Path, default=base_data / "eda_profile.md")
    parser.add_argument("--output-html", type=Path, default=base_data / "eda_profile.html")
    parser.add_argument("--html", action="store_true", help="Also generate HTML profile report (ydata-profiling)")
    args = parser.parse_args()

    cleaned = safe_read_parquet_or_csv(args.cleaned_parquet, args.cleaned_csv)

    exploded: Optional[pd.DataFrame] = None
    try:
        exploded = safe_read_parquet_or_csv(args.exploded_parquet, args.exploded_csv)
    except FileNotFoundError:
        exploded = None

    report_md = build_report(cleaned, exploded)
    args.output_md.write_text(report_md, encoding="utf-8")

    if args.html:
        make_profile_report(cleaned, args.output_html)

    print({
        "markdown": str(args.output_md),
        "html": str(args.output_html) if args.html else None,
    })


if __name__ == "__main__":
    main()


