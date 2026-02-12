"""
Poisson playoff "test model" using team-level international stats.

- Reads cleaned CSVs from a folder (your cleaned_soccer_data folder)
- Filters to ONLY the countries you care about
- Builds attack/defense strengths from goals per match
- Uses Poisson to estimate win/draw/loss probabilities for neutral matches
- Runs a neutral round-robin inside each group and picks the top team
"""

from __future__ import annotations

import os
import glob
import math
from typing import Dict, List, Tuple

import pandas as pd


# =========================
# 1) CONFIG
# =========================

CLEAN_DIR = os.path.expanduser("~/Downloads/cleaned_soccer_data")

GROUPS: Dict[str, List[str]] = {
    "G1": ["DEN", "MKD", "CZE", "IRL"],
    "G2": ["ITA", "NIR", "WAL", "BIH"],
    "G3": ["TUR", "ROU", "SVK", "KOS"],
    "G4": ["UKR", "SWE", "POL", "ALB"],
    "G5": ["BOL", "SUR", "IRQ"],
    "G6": ["NCL", "JAM", "COD"],
}

# If your CSV's `country` column is NOT 3-letter codes, fill this mapping.
# Example:
# CODE_TO_COUNTRY = {
#     "DEN": "Denmark",
#     "MKD": "North Macedonia",
#     ...
# }
CODE_TO_COUNTRY = {
    # Group 1
    "DEN": "Denmark",
    "MKD": "FYR Macedonia",
    "CZE": "Czech Republic",
    "IRL": "Republic of Ireland",

    # Group 2
    "ITA": "Italy",
    "NIR": "Northern Ireland",
    "WAL": "Wales",
    "BIH": "Bosnia and Herzegovina",

    # Group 3
    "TUR": "Turkey",
    "ROU": "Romania",
    "SVK": "Slovakia",
    "KOS": "Kosovo",

    # Group 4
    "UKR": "Ukraine",
    "SWE": "Sweden",
    "POL": "Poland",
    "ALB": "Albania",

    # Group 5
    "BOL": "Bolivia",
    "SUR": "Suriname",
    "IRQ": "Iraq",

    # Group 6
    "NCL": "New Caledonia",
    "JAM": "Jamaica",
    "COD": "Congo DR",
}

# Poisson summation cutoff. Higher = more accurate but slower.
MAX_GOALS = 10


# =========================
# 2) LOAD + FILTER
# =========================

def _is_probably_code_series(s: pd.Series) -> bool:
    """Heuristic: checks if values look like 3-letter uppercase codes."""
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    return sample.map(lambda x: len(x) == 3 and x.isupper()).mean() > 0.8


def load_cleaned_team_data(clean_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(clean_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {clean_dir}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file__"] = os.path.basename(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Normalize column names: your cleaned data uses these exact names
    needed = {
        "country",
        "matches_played",
        "goals_scored",
        "goals_conceded",
        "goals_scored_per_match",
        "goals_conceded_per_match",
    }
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns in cleaned data: "
            f"{sorted(missing)}\n"
            "Your cleaner must keep goals + matches columns."
        )

    return data


def restrict_to_countries(data: pd.DataFrame, country_codes: List[str]) -> pd.DataFrame:
    """
    Filters data to only the countries in `country_codes`.

    Works whether data['country'] contains:
      - 3-letter codes (DEN, ITA, ...)
      - full names (Denmark, Italy, ...) via CODE_TO_COUNTRY mapping
    """
    country_col = data["country"].astype(str)

    # Case A: country column already looks like 3-letter codes
    if _is_probably_code_series(country_col):
        mask = country_col.isin(country_codes)
        out = data[mask].copy()
        out["country_code"] = out["country"]
        return out

    # Case B: country column looks like names; require mapping
    if not CODE_TO_COUNTRY:
        raise ValueError(
            "Your CSV `country` column does not look like 3-letter codes.\n"
            "Fill CODE_TO_COUNTRY with mappings like {'DEN': 'Denmark', ...}."
        )

    wanted_names = {CODE_TO_COUNTRY[c] for c in country_codes if c in CODE_TO_COUNTRY}
    out = data[country_col.isin(wanted_names)].copy()

    # Add country_code by reverse mapping
    name_to_code = {v: k for k, v in CODE_TO_COUNTRY.items()}
    out["country_code"] = out["country"].map(name_to_code)

    # Drop any rows that didn't map cleanly
    out = out.dropna(subset=["country_code"])
    return out


# =========================
# 3) BUILD TEAM STRENGTHS
# =========================

def aggregate_country_stats(country_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates across competitions/seasons for each country_code.

    We aggregate totals using:
      - goals_scored, goals_conceded, matches_played
    then compute goals per match.
    """
    # Ensure numeric
    for col in ["matches_played", "goals_scored", "goals_conceded"]:
        country_data[col] = pd.to_numeric(country_data[col], errors="coerce")

    agg = (
        country_data
        .dropna(subset=["country_code", "matches_played", "goals_scored", "goals_conceded"])
        .groupby("country_code", as_index=False)
        .agg(
            matches_played=("matches_played", "sum"),
            goals_scored=("goals_scored", "sum"),
            goals_conceded=("goals_conceded", "sum"),
        )
    )

    # Prevent division by zero
    agg = agg[agg["matches_played"] > 0].copy()
    agg["gf_per_match"] = agg["goals_scored"] / agg["matches_played"]
    agg["ga_per_match"] = agg["goals_conceded"] / agg["matches_played"]

    return agg


def compute_attack_defense(agg: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Poisson-style strengths:
      attack = (team gf/match) / (overall gf/match)
      defense = (team ga/match) / (overall ga/match)

    We use overall_avg_goals as the baseline lambda.
    """
    overall_avg_goals = agg["gf_per_match"].mean()

    # If overall goals per match is 0 (shouldn't happen), fallback
    if overall_avg_goals <= 0:
        overall_avg_goals = 1.0

    # For defense scaling baseline, use overall ga/match mean
    overall_avg_concede = agg["ga_per_match"].mean()
    if overall_avg_concede <= 0:
        overall_avg_concede = overall_avg_goals

    out = agg.copy()
    out["attack"] = out["gf_per_match"] / overall_avg_goals
    out["defense"] = out["ga_per_match"] / overall_avg_concede

    return out, overall_avg_goals


# =========================
# 4) POISSON MATCH PROBS
# =========================

def poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def wdl_probs(lam_a: float, lam_b: float, max_goals: int = MAX_GOALS) -> Tuple[float, float, float]:
    """
    Returns (P(A wins), P(draw), P(B wins)) for a neutral match:
      Goals_A ~ Poisson(lam_a), Goals_B ~ Poisson(lam_b)
    """
    pa = [poisson_pmf(lam_a, i) for i in range(max_goals + 1)]
    pb = [poisson_pmf(lam_b, j) for j in range(max_goals + 1)]

    p_awin = 0.0
    p_draw = 0.0
    p_bwin = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pa[i] * pb[j]
            if i > j:
                p_awin += p
            elif i == j:
                p_draw += p
            else:
                p_bwin += p

    # (Tiny leftover mass beyond max_goals is ignored; increase MAX_GOALS if desired.)
    return p_awin, p_draw, p_bwin


def expected_lambdas(team_a: str, team_b: str, strengths: pd.DataFrame, base_goals: float) -> Tuple[float, float]:
    """
    Neutral-site expected goals:
      λ_A = base_goals * attack_A * defense_B
      λ_B = base_goals * attack_B * defense_A
    """
    row_a = strengths.loc[strengths["country_code"] == team_a].iloc[0]
    row_b = strengths.loc[strengths["country_code"] == team_b].iloc[0]

    lam_a = base_goals * float(row_a["attack"]) * float(row_b["defense"])
    lam_b = base_goals * float(row_b["attack"]) * float(row_a["defense"])

    # Safety clamps
    lam_a = max(lam_a, 0.01)
    lam_b = max(lam_b, 0.01)
    return lam_a, lam_b


# =========================
# 5) GROUP ROUND-ROBIN + PICK WINNER
# =========================

def expected_points_round_robin(teams: List[str], strengths: pd.DataFrame, base_goals: float) -> pd.DataFrame:
    """
    Each pair plays once (neutral). Expected points:
      E[pts] = 3*P(win) + 1*P(draw)
    """
    pts = {t: 0.0 for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            lam_a, lam_b = expected_lambdas(a, b, strengths, base_goals)
            p_awin, p_draw, p_bwin = wdl_probs(lam_a, lam_b)

            pts[a] += 3 * p_awin + 1 * p_draw
            pts[b] += 3 * p_bwin + 1 * p_draw

    out = pd.DataFrame({"team": list(pts.keys()), "expected_points": list(pts.values())})
    out = out.sort_values(["expected_points", "team"], ascending=[False, True]).reset_index(drop=True)
    return out


def main() -> None:
    all_codes = sorted({c for group in GROUPS.values() for c in group})

    data = load_cleaned_team_data(CLEAN_DIR)
    data = restrict_to_countries(data, all_codes)

    agg = aggregate_country_stats(data)
    strengths, base_goals = compute_attack_defense(agg)

    # Ensure we actually have every team
    have = set(strengths["country_code"].tolist())
    missing = [c for c in all_codes if c not in have]
    if missing:
        raise ValueError(
            "Missing teams in your cleaned dataset after filtering:\n"
            f"{missing}\n"
            "Either those countries never appear in your CSVs, or your CODE_TO_COUNTRY mapping is incomplete."
        )

    print("\n=== Base goals per match (league avg) ===")
    print(f"{base_goals:.3f}\n")

    winners: Dict[str, str] = {}

    for gname, teams in GROUPS.items():
        table = expected_points_round_robin(teams, strengths, base_goals)
        winner = table.iloc[0]["team"]
        winners[gname] = str(winner)

        print(f"=== {gname}: {teams} ===")
        print(table.to_string(index=False))
        print(f"-> PICK: {winner}\n")

    print("=== FINAL PICKS (one per group) ===")
    for gname in sorted(winners.keys()):
        print(f"{gname}: {winners[gname]}")


if __name__ == "__main__":
    main()