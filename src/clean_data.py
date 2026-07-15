from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fiverr-data-gigs.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fiverr_gigs_cleaned.csv"
)


def parse_rating(value):
    if pd.isna(value):
        return np.nan

    match = re.match(
        r"^\s*(\d+(?:\.\d+)?)",
        str(value),
    )

    return (
        float(match.group(1))
        if match
        else np.nan
    )


def parse_review_count(value):
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"\(([^)]+)\)",
        str(value),
    )

    if not match:
        return np.nan

    text = (
        match.group(1)
        .lower()
        .replace(",", "")
        .replace("+", "")
        .strip()
    )

    multiplier = 1

    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]

    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def parse_price(value):
    if pd.isna(value):
        return np.nan

    text = re.sub(
        r"[^\d]",
        "",
        str(value),
    )

    if text == "":
        return np.nan

    return float(text)


def clean_data(df_raw):
    df = df_raw.rename(
        columns={
            "Title": "gig_title",
            "Title_URL": "gig_url",
            "gigrating": "rating_raw",
            "_5fo9i5": "seller_level_raw",
            "Price": "price_raw",
        }
    ).copy()

    df["rating"] = (
        df["rating_raw"]
        .apply(parse_rating)
    )

    df["review_count"] = (
        df["rating_raw"]
        .apply(parse_review_count)
        .astype("Int64")
    )

    df["price_pkr"] = (
        df["price_raw"]
        .apply(parse_price)
        .astype("Int64")
    )

    valid_levels = {
        "Level 1 Seller",
        "Level 2 Seller",
        "Top Rated Seller",
    }

    df["seller_level"] = (
        df["seller_level_raw"]
        .where(
            df["seller_level_raw"]
            .isin(valid_levels),
            "Unknown",
        )
    )

    df["title_clean"] = (
        df["gig_title"]
        .astype(str)
        .str.lower()
        .str.replace(
            r"[^a-z0-9\s]",
            " ",
            regex=True,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    df["title_word_count"] = (
        df["title_clean"]
        .str.split()
        .str.len()
    )

    df["seller_username"] = (
        df["gig_url"]
        .str.extract(
            r"fiverr\.com/([^/]+)/",
            expand=False,
        )
    )

    df["listing_position"] = pd.to_numeric(
        df["gig_url"].str.extract(
            r"[?&]pos=(\d+)",
            expand=False,
        ),
        errors="coerce",
    )

    df = (
        df.drop_duplicates(
            subset=["gig_url"]
        )
        .reset_index(drop=True)
    )

    df["log_price_pkr"] = np.log1p(
        df["price_pkr"].astype(float)
    )

    df["log_review_count"] = np.log1p(
        df["review_count"].astype(float)
    )

    return df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {RAW_PATH}"
        )

    print("Membaca:", RAW_PATH)

    df_raw = pd.read_csv(RAW_PATH)

    print(
        "Ukuran data mentah:",
        df_raw.shape,
    )

    df_clean = clean_data(df_raw)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_clean.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Ukuran data bersih:",
        df_clean.shape,
    )

    print(
        "Data disimpan ke:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()