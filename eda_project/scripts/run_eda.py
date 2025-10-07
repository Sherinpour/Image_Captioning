import argparse
import io
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024**2)


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    info_str = info_buf.getvalue()
    overview = {
        "num_rows": int(len(df)),
        "num_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "memory_usage_mb": round(memory_usage_mb(df), 2),
        "info": info_str,
    }
    return overview


def describe_numerics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    desc = numeric_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    return desc


def top_frequencies(series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    vc = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .value_counts()
        .head(top_n)
    )
    return vc.rename_axis("value").reset_index(name="count")


def try_parse_json_cell(cell: object) -> Optional[object]:
    if isinstance(cell, (list, dict)):
        return cell
    if isinstance(cell, str):
        s = cell.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def extract_entities(df: pd.DataFrame, entities_col: str = "entities") -> pd.DataFrame:
    if entities_col not in df.columns:
        return pd.DataFrame(columns=["entity_name", "entity_value"])

    parsed: List[Tuple[str, str]] = []
    for val in df[entities_col].tolist():
        parsed_val = try_parse_json_cell(val)
        if isinstance(parsed_val, list):
            for item in parsed_val:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("entity") or "").strip()
                    value = str(item.get("value") or item.get("val") or "").strip()
                    if name or value:
                        parsed.append((name, value))
        elif isinstance(parsed_val, dict):
            for k, v in parsed_val.items():
                name = str(k).strip()
                value = str(v).strip()
                if name or value:
                    parsed.append((name, value))

    ents_df = pd.DataFrame(parsed, columns=["entity_name", "entity_value"])
    return ents_df


# --- Persian text utilities ---

PERSIAN_YE = "ی"
ARABIC_YE = "ي"
PERSIAN_KEHEH = "ک"
ARABIC_KAF = "ك"
DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_LETTER_RE = re.compile(r"[^\u0600-\u06FF\s]+")


def normalize_persian(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = s.replace(ARABIC_YE, PERSIAN_YE).replace(ARABIC_KAF, PERSIAN_KEHEH)
    s = DIACRITICS.sub("", s)
    s = NON_LETTER_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


DEFAULT_STOPWORDS = {
    # Common Persian/Arabic stopwords (minimal, extend as needed)
    "و",
    "در",
    "به",
    "از",
    "که",
    "با",
    "برای",
    "این",
    "آن",
    "های",
    "یا",
    "تا",
    "یک",
    "می",
    "است",
    "شود",
    "کرد",
    "کردن",
    "را",
    "بر",
    "هم",
    "اما",
    "بدون",
}


def tokenize_persian(text: str, stopwords: Optional[set] = None) -> List[str]:
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    norm = normalize_persian(text)
    if not norm:
        return []
    tokens = [t for t in norm.split(" ") if t and t not in stopwords]
    return tokens


def top_words(series: pd.Series, top_n: int = 50) -> pd.DataFrame:
    counter: Counter = Counter()
    for val in series.dropna().astype(str).tolist():
        tokens = tokenize_persian(val)
        counter.update(tokens)
    most = counter.most_common(top_n)
    return pd.DataFrame(most, columns=["token", "count"])


def safe_imports_for_wordcloud():
    try:
        import arabic_reshaper  # noqa: F401
        from bidi.algorithm import get_display  # noqa: F401
        from wordcloud import WordCloud  # noqa: F401

        return True
    except Exception:
        return False


def build_persian_wordcloud(
    text: str, out_path: Path, width: int = 1200, height: int = 600
) -> Optional[Path]:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        from wordcloud import WordCloud

        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        wc = WordCloud(
            font_path=None, width=width, height=height, background_color="white"
        )
        img = wc.generate(bidi_text).to_image()
        img.save(out_path)
        return out_path
    except Exception:
        return None


def series_to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    limited = df.head(max_rows)
    return limited.to_markdown(index=False)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return pd.DataFrame()
    corr = numeric_df.corr(numeric_only=True)
    return corr


def save_heatmap(corr: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if corr.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_barplot(
    counts: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path
) -> Optional[Path]:
    if counts.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    sns.barplot(data=counts, x=x_col, y=y_col, color="#4C78A8")
    plt.title(title)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_missingno_plots(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    try:
        import missingno as msno
        import matplotlib.pyplot as plt

        # Matrix
        plt.figure(figsize=(12, 6))
        msno.matrix(
            df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df
        )
        p = out_dir / "missing_matrix.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)

        # Bar
        plt.figure(figsize=(12, 6))
        msno.bar(df)
        p = out_dir / "missing_bar.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    except Exception:
        pass
    return paths


def generate_profile_report(
    df: pd.DataFrame, out_html: Path, minimal: bool = True, sample_rows: int = 10000
) -> Optional[Path]:
    try:
        try:
            from ydata_profiling import ProfileReport  # type: ignore
        except Exception:
            from pandas_profiling import ProfileReport  # type: ignore

        prof_df = df
        if sample_rows and len(df) > sample_rows:
            prof_df = df.sample(sample_rows, random_state=42)
        profile = ProfileReport(prof_df, title="EDA Profile Report", minimal=minimal)
        profile.to_file(out_html)
        return out_html
    except Exception:
        return None


def build_markdown_report(
    overview: Dict[str, object],
    numerics_desc: pd.DataFrame,
    group_freq: pd.DataFrame,
    product_freq: pd.DataFrame,
    entity_name_freq: pd.DataFrame,
    entity_value_freq: pd.DataFrame,
    title_top_words: pd.DataFrame,
    corr_heatmap_path: Optional[Path],
    missing_paths: List[Path],
    wordcloud_path: Optional[Path],
) -> str:
    md_parts: List[str] = []
    md_parts.append("## Data Overview")
    md_parts.append(f"- **Rows**: {overview['num_rows']}")
    md_parts.append(f"- **Columns**: {overview['num_cols']}")
    md_parts.append(f"- **Memory usage (MB)**: {overview['memory_usage_mb']}")
    md_parts.append("")
    md_parts.append("### Missing values by column")
    miss_series = pd.Series(overview["missing_per_column"])  # type: ignore
    md_parts.append(
        miss_series.sort_values(ascending=False).to_frame("missing").to_markdown()
    )

    md_parts.append("")
    md_parts.append("## Descriptive Statistics (Numerical)")
    if numerics_desc.empty:
        md_parts.append("No numerical columns detected.")
    else:
        md_parts.append(numerics_desc.to_markdown())

    md_parts.append("")
    md_parts.append("## Key Frequencies")
    if not group_freq.empty:
        md_parts.append("### Top groups")
        md_parts.append(group_freq.to_markdown(index=False))
    if not product_freq.empty:
        md_parts.append("\n### Top products")
        md_parts.append(product_freq.to_markdown(index=False))
    if not entity_name_freq.empty:
        md_parts.append("\n### Top entity names")
        md_parts.append(entity_name_freq.to_markdown(index=False))
    if not entity_value_freq.empty:
        md_parts.append("\n### Top entity values")
        md_parts.append(entity_value_freq.to_markdown(index=False))

    md_parts.append("")
    md_parts.append("## Relationships and Correlations")
    if corr_heatmap_path is not None:
        md_parts.append(f"![Correlation Heatmap]({corr_heatmap_path.as_posix()})")
    else:
        md_parts.append("No numeric correlations available.")

    if missing_paths:
        md_parts.append("\n## Missing Data Visualizations")
        for p in missing_paths:
            md_parts.append(f"![Missingness]({p.as_posix()})")

    md_parts.append("\n## Persian Text Analysis")
    if not title_top_words.empty:
        md_parts.append("### Top tokens in title")
        md_parts.append(title_top_words.to_markdown(index=False))
    if wordcloud_path is not None:
        md_parts.append(f"\n![Title Wordcloud]({wordcloud_path.as_posix()})")

    md_parts.append("\n## Notes")
    md_parts.append("- Frequency tables are top-20 by count unless otherwise noted.")
    md_parts.append(
        "- Text normalized for Persian orthography; minimal stopword removal applied."
    )

    return "\n\n".join(md_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EDA on main_entities_dataframe.csv and produce Markdown/HTML outputs"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "main_entities_dataframe.csv",
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).resolve().parent / "EDA_Report.md",
        help="Path to the Markdown report to write",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path(__file__).resolve().parent / "EDA_Profile_Report.html",
        help="Optional path to write a profiling HTML report",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable HTML profiling report generation",
    )
    args = parser.parse_args()

    df = read_csv_safely(args.csv)

    # Derive helper numeric features (e.g., text lengths) for correlations
    for col in ("title", "product", "group"):
        if col in df.columns:
            df[f"{col}_length"] = df[col].astype(str).str.len()

    overview = summarize_dataframe(df)
    numerics_desc = describe_numerics(df)

    group_freq = (
        top_frequencies(df["group"]) if "group" in df.columns else pd.DataFrame()
    )
    product_freq = (
        top_frequencies(df["product"]) if "product" in df.columns else pd.DataFrame()
    )

    ents_df = extract_entities(df, entities_col="entities")
    entity_name_freq = (
        top_frequencies(ents_df["entity_name"]) if not ents_df.empty else pd.DataFrame()
    )
    entity_value_freq = (
        top_frequencies(ents_df["entity_value"])
        if not ents_df.empty
        else pd.DataFrame()
    )

    corr_df = compute_correlations(df)

    safe_mkdir(args.fig_dir)
    corr_path = None
    if not corr_df.empty:
        corr_path = save_heatmap(corr_df, args.fig_dir / "correlation_heatmap.png")

    # Visualize top categories
    group_bar = None
    if not group_freq.empty:
        group_bar = save_barplot(
            group_freq,
            x_col="value",
            y_col="count",
            title="Top Groups",
            out_path=args.fig_dir / "top_groups.png",
        )
    product_bar = None
    if not product_freq.empty:
        product_bar = save_barplot(
            product_freq,
            x_col="value",
            y_col="count",
            title="Top Products",
            out_path=args.fig_dir / "top_products.png",
        )

    # Missingness plots
    missing_paths = save_missingno_plots(df, args.fig_dir)

    # Persian text analysis on title
    title_top = top_words(df["title"]) if "title" in df.columns else pd.DataFrame()
    wc_path = None
    if safe_imports_for_wordcloud() and not title_top.empty:
        # Build full text from most frequent tokens (repeated by count for visual emphasis)
        text_blob = " ".join(
            [token for token, count in title_top.values for _ in range(int(count))]
        )
        wc_path = build_persian_wordcloud(
            text_blob, args.fig_dir / "title_wordcloud.png"
        )

    # Build Markdown report
    md = build_markdown_report(
        overview=overview,
        numerics_desc=numerics_desc,
        group_freq=group_freq,
        product_freq=product_freq,
        entity_name_freq=entity_name_freq,
        entity_value_freq=entity_value_freq,
        title_top_words=title_top,
        corr_heatmap_path=corr_path,
        missing_paths=missing_paths,
        wordcloud_path=wc_path,
    )

    # Save outputs
    safe_mkdir(args.out_md.parent)
    args.out_md.write_text(md, encoding="utf-8")

    if not args.no_profile:
        _ = generate_profile_report(df, args.out_html, minimal=True)

    print(
        {
            "markdown_report": str(args.out_md),
            "html_profile": None if args.no_profile else str(args.out_html),
            "figures_dir": str(args.fig_dir),
        }
    )


if __name__ == "__main__":
    main()
