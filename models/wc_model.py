from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================
# 1) GROUPS (YOUR INPUT)
# =========================

GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Denmark"],
    "B": ["Canada", "Switzerland", "Qatar", "Italy"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Australia", "Paraguay", "Slovakia"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Poland"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Congo DR"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

N_SIMS = 20000  # increase if you want tighter estimates

# =========================
# 2) DATA: LOAD FROM FOLDER + USE ALL COLUMNS (NON-COLLINEAR)
# =========================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(REPO_ROOT, "data", "cleaned_soccer_data")

REQUIRED_COLS = {"country", "matches_played", "goals_scored", "goals_conceded"}

COLLINEAR_CORR = 0.95
MAX_MISSING_FRAC = 0.50


def load_all_csvs_from_folder(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file__"] = os.path.basename(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def choose_features_all_numeric(df: pd.DataFrame) -> List[str]:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target + direct target transforms / leakage
    drop = {
        "matches_played",
        "goals_scored",
        "goals_conceded",
        "goal_difference",
        "total_goal_count",
        "goals_scored_per_match",
        "goals_conceded_per_match",
    }

    feats = [c for c in numeric_cols if c not in drop]

    if feats:
        miss_frac = df[feats].isna().mean()
        feats = [c for c in feats if miss_frac[c] < MAX_MISSING_FRAC]

        nunique = df[feats].nunique(dropna=True)
        feats = [c for c in feats if nunique[c] > 1]

    if not feats:
        return []

    X = df[feats].fillna(df[feats].median(numeric_only=True))

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if any(upper[col] >= COLLINEAR_CORR):
            to_drop.add(col)

    return [c for c in feats if c not in to_drop]

# =========================
# FIFA + CONFED WEIGHTS
# =========================

# Wikipedia module that mirrors FIFA ranking points and is easy to parse.
# It lists (team_name, rank, movement, points).
FIFA_RANKINGS_RAW_URL = (
    "https://en.wikipedia.org/wiki/Module:SportsRankings/data/FIFA_World_Rankings?action=raw"
)

# Your GROUPS use different names for a few teams vs FIFA/Wikipedia.
FIFA_NAME_ALIASES = {
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "New Zealand": "Aotearoa New Zealand",
}

# Confederation mapping for the 48 teams in your GROUPS
TEAM_CONFED = {
    # CONCACAF
    "Mexico": "CONCACAF",
    "Canada": "CONCACAF",
    "USA": "CONCACAF",
    "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF",
    "Panama": "CONCACAF",

    # UEFA
    "Denmark": "UEFA",
    "Switzerland": "UEFA",
    "Italy": "UEFA",
    "Scotland": "UEFA",
    "Slovakia": "UEFA",
    "Germany": "UEFA",
    "Netherlands": "UEFA",
    "Poland": "UEFA",
    "Belgium": "UEFA",
    "Spain": "UEFA",
    "France": "UEFA",
    "Norway": "UEFA",
    "Austria": "UEFA",
    "Portugal": "UEFA",
    "England": "UEFA",
    "Croatia": "UEFA",

    # CONMEBOL
    "Brazil": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Argentina": "CONMEBOL",
    "Colombia": "CONMEBOL",

    # AFC
    "South Korea": "AFC",
    "Qatar": "AFC",
    "Japan": "AFC",
    "Iran": "AFC",
    "Saudi Arabia": "AFC",
    "Iraq": "AFC",
    "Jordan": "AFC",
    "Uzbekistan": "AFC",

    # CAF
    "South Africa": "CAF",
    "Morocco": "CAF",
    "Tunisia": "CAF",
    "Egypt": "CAF",
    "Senegal": "CAF",
    "Algeria": "CAF",
    "Congo DR": "CAF",
    "Ghana": "CAF",
    "Cape Verde": "CAF",

    # OFC
    "New Zealand": "OFC",

    # (Australia is AFC in FIFA)
    "Australia": "AFC",
}

def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8", errors="replace")

def fetch_fifa_points_map() -> Dict[str, float]:
    """
    Returns dict: FIFA-style team name -> FIFA points (float).
    Source contains lines like: {  "Spain", 1, 0, 1877.18 },
    """
    raw = _fetch_text(FIFA_RANKINGS_RAW_URL)
    # Capture team + points (ignore rank/movement)
    pattern = r'\{\s*"([^"]+)"\s*,\s*\d+\s*,\s*-?\d+\s*,\s*([0-9]+\.[0-9]+)\s*\}'
    pairs = re.findall(pattern, raw)
    return {name: float(pts) for name, pts in pairs}

def team_fifa_points(team: str, points_map: Dict[str, float], default: float) -> float:
    lookup = FIFA_NAME_ALIASES.get(team, team)
    return float(points_map.get(lookup, default))

def build_confed_strengths_from_points(points_by_team: Dict[str, float]) -> Dict[str, float]:
    """
    Computes a confederation multiplier from the average FIFA points of teams
    in that confed (only among your 48 teams).
    Normalizes so overall mean confed weight = 1.0.
    """
    confed_vals: Dict[str, List[float]] = {}
    for t, pts in points_by_team.items():
        conf = TEAM_CONFED.get(t)
        if conf is None:
            continue
        confed_vals.setdefault(conf, []).append(pts)

    confed_mean = {c: (sum(v) / len(v)) for c, v in confed_vals.items() if v}
    overall = sum(confed_mean.values()) / len(confed_mean) if confed_mean else 1.0

    # multiplier: confed avg / overall avg
    return {c: (m / overall) for c, m in confed_mean.items()}

def get_year_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Try to find a usable 'year' column. Adjust these candidates if your data uses
    a different name.
    """
    for col in ["year", "Year", "season_year", "tournament_year"]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().any():
                return y
    return None

def recency_weights(years: pd.Series, half_life_years: float = 6.0) -> pd.Series:
    """
    Exponential decay so recent rows count more.
    half_life_years=6 means data 6 years old counts ~50% as much.
    """
    y = years.copy()
    max_y = float(y.max())
    # decay constant: w = 0.5 ** (age / half_life)
    age = (max_y - y).clip(lower=0)
    return (0.5 ** (age / half_life_years)).fillna(1.0)


def fit_poisson_glm(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: List[str],
    weights: pd.Series | None = None,
) -> sm.GLM:
    d = df.copy()
    d = d[d["matches_played"] > 0].copy()

    y = pd.to_numeric(d[y_col], errors="coerce")
    d = d.loc[y.notna()].copy()
    y = y.loc[d.index]

    if feature_cols:
        X = d[feature_cols].copy().fillna(d[feature_cols].median(numeric_only=True))
        X = sm.add_constant(X, has_constant="add")
    else:
        X = sm.add_constant(pd.DataFrame(index=d.index), has_constant="add")

    offset = np.log(d["matches_played"].astype(float))

    if weights is not None:
        w = pd.to_numeric(weights.loc[d.index], errors="coerce").fillna(1.0)
        w = w.clip(lower=0.0)
    else:
        w = None

    model = sm.GLM(
        y, X,
        family=sm.families.Poisson(),
        offset=offset,
        freq_weights=w,
    )
    return model.fit()


def build_strengths_from_all_columns(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, float, List[str]]:
    feature_cols = choose_features_all_numeric(df_all)

    # --- FIFA points for the 48 teams in your GROUPS ---
    all_group_teams = sorted({t for teams in GROUPS.values() for t in teams})

    try:
        points_map = fetch_fifa_points_map()
        default_pts = float(np.median(list(points_map.values())))
    except Exception:
        points_map = {}
        default_pts = 1500.0  # reasonable fallback

    points_by_team = {t: team_fifa_points(t, points_map, default_pts) for t in all_group_teams}
    confed_strength = build_confed_strengths_from_points(points_by_team)

    # --- Build per-row weights for your training data ---
    # team factor from FIFA points (mild effect so it doesn’t overpower your data)
    # normalized around 1.0
    pts_mean = float(np.mean(list(points_by_team.values()))) if points_by_team else default_pts
    team_pts_scale = {t: (points_by_team[t] / pts_mean) for t in all_group_teams}

    # If your df_all has a country column, map weights by row-country.
    # If a country isn't in your 48-team list, weight=1.0.
    def row_weight_for_country(c: str) -> float:
        # confed multiplier
        conf = TEAM_CONFED.get(c)
        w_conf = confed_strength.get(conf, 1.0) if conf else 1.0
        # fifa points multiplier
        w_pts = team_pts_scale.get(c, 1.0)
        return float(w_conf * w_pts)

    base_w = df_all["country"].map(row_weight_for_country).fillna(1.0)

    # Optional: year-based recency weights
    y_series = get_year_series(df_all)
    if y_series is not None:
        w_time = recency_weights(y_series, half_life_years=6.0)
    else:
        w_time = pd.Series(1.0, index=df_all.index)

    weights = (base_w * w_time).astype(float)

    # --- Fit weighted GLMs ---
    atk_fit = fit_poisson_glm(df_all, "goals_scored", feature_cols, weights=weights)
    def_fit = fit_poisson_glm(df_all, "goals_conceded", feature_cols, weights=weights)

    d = df_all.copy()
    d = d[d["matches_played"] > 0].copy()

    if feature_cols:
        X = d[feature_cols].copy().fillna(d[feature_cols].median(numeric_only=True))
        X = sm.add_constant(X, has_constant="add")
    else:
        X = sm.add_constant(pd.DataFrame(index=d.index), has_constant="add")

    offset = np.log(d["matches_played"].astype(float))

    mu_gf_total = atk_fit.predict(X, offset=offset)
    mu_ga_total = def_fit.predict(X, offset=offset)

    pred = d[["country", "matches_played"]].copy()
    pred["mu_gf_total"] = mu_gf_total
    pred["mu_ga_total"] = mu_ga_total

    team = (
        pred.groupby("country", as_index=True)
        .agg(
            matches_played=("matches_played", "sum"),
            mu_gf_total=("mu_gf_total", "sum"),
            mu_ga_total=("mu_ga_total", "sum"),
        )
    )

    team = team[team["matches_played"] > 0].copy()
    team["gf_per_match"] = team["mu_gf_total"] / team["matches_played"]
    team["ga_per_match"] = team["mu_ga_total"] / team["matches_played"]

    base_goals = float(team["gf_per_match"].mean())
    if base_goals <= 0:
        base_goals = 1.0

    team["attack"] = team["gf_per_match"] / base_goals
    team["defense"] = team["ga_per_match"] / base_goals

    return team[["attack", "defense"]].copy(), float(base_goals), feature_cols

# =========================
# 3) POISSON MATCH SIMULATION
# =========================

def poisson_sample(lam: float) -> int:
    return int(np.random.poisson(max(lam, 0.01)))


def expected_lambdas(team_a: str, team_b: str) -> Tuple[float, float]:
    if team_a in strengths.index:
        a_attack = float(strengths.loc[team_a, "attack"])
        a_def = float(strengths.loc[team_a, "defense"])
    else:
        a_attack, a_def = 1.0, 1.0

    if team_b in strengths.index:
        b_attack = float(strengths.loc[team_b, "attack"])
        b_def = float(strengths.loc[team_b, "defense"])
    else:
        b_attack, b_def = 1.0, 1.0

    lam_a = BASE_GOALS * a_attack * b_def
    lam_b = BASE_GOALS * b_attack * a_def
    return float(max(lam_a, 0.01)), float(max(lam_b, 0.01))


def play_match(team_a: str, team_b: str) -> Tuple[int, int]:
    lam_a, lam_b = expected_lambdas(team_a, team_b)
    return poisson_sample(lam_a), poisson_sample(lam_b)


# =========================
# 4) GROUP STAGE
# =========================

@dataclass
class TableRow:
    team: str
    pts: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def simulate_group(group_teams: List[str]) -> List[TableRow]:
    rows = {t: TableRow(team=t) for t in group_teams}

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            a, b = group_teams[i], group_teams[j]
            ga, gb = play_match(a, b)

            rows[a].gf += ga
            rows[a].ga += gb
            rows[b].gf += gb
            rows[b].ga += ga

            if ga > gb:
                rows[a].pts += 3
            elif gb > ga:
                rows[b].pts += 3
            else:
                rows[a].pts += 1
                rows[b].pts += 1

    table = list(rows.values())
    table.sort(key=lambda r: (r.pts, r.gd, r.gf, r.team), reverse=True)
    return table


# =========================
# 5) FIXED (NON-RANDOM) KNOCKOUT HELPERS + VERBOSE MATCH PRINTING
# =========================

def rank_row(r: TableRow) -> Tuple[int, int, int, str]:
    return (r.pts, r.gd, r.gf, r.team)


def select_qualifiers(
    group_tables: Dict[str, List[TableRow]]
) -> Tuple[
    List[Tuple[str, TableRow]],
    List[Tuple[str, TableRow]],
    List[Tuple[str, TableRow]]
]:
    winners = [(g, group_tables[g][0]) for g in GROUPS.keys()]
    runners = [(g, group_tables[g][1]) for g in GROUPS.keys()]
    thirds  = [(g, group_tables[g][2]) for g in GROUPS.keys()]
    return winners, runners, thirds


def pick_best_third_place(
    thirds: List[Tuple[str, TableRow]],
    k: int = 8
) -> List[Tuple[str, TableRow]]:
    thirds_sorted = sorted(thirds, key=lambda gr: rank_row(gr[1]), reverse=True)
    return thirds_sorted[:k]

def penalty_prob(team_a: str, team_b: str) -> float:
    lam_a, lam_b = expected_lambdas(team_a, team_b)

    # baseline probability from relative strength
    p = lam_a / (lam_a + lam_b)

    # shrink toward 50/50 so shootouts are still noisy (very important)
    p = 0.5 + 0.35 * (p - 0.5)   # 0.35 controls how “skill-based” pens are

    # clamp to avoid extremes
    return float(min(0.65, max(0.35, p)))


def penalty_winner(team_a: str, team_b: str) -> str:
    return team_a if random.random() < penalty_prob(team_a, team_b) else team_b

# --- FIFA ROUND OF 32 BRACKET (MATCHES 73–88) ---

# These are the 8 matches in the Round of 32 that involve a 3rd-place qualifier,
# and which groups that 3rd-place team is allowed to come from.
THIRD_SLOT_ALLOWED_GROUPS: Dict[int, set[str]] = {
    74: {"A", "B", "C", "D", "F"},          # Winner E vs 3rd A/B/C/D/F
    77: {"C", "D", "F", "G", "H"},          # Winner I vs 3rd C/D/F/G/H
    79: {"C", "E", "F", "H", "I"},          # Winner A vs 3rd C/E/F/H/I
    80: {"E", "H", "I", "J", "K"},          # Winner L vs 3rd E/H/I/J/K
    81: {"B", "E", "F", "I", "J"},          # Winner D vs 3rd B/E/F/I/J
    82: {"A", "E", "H", "I", "J"},          # Winner G vs 3rd A/E/H/I/J
    85: {"E", "F", "G", "I", "J"},          # Winner B vs 3rd E/F/G/I/J
    87: {"D", "E", "I", "J", "L"},          # Winner K vs 3rd D/E/I/J/L
}

THIRD_MATCH_ORDER = [74, 77, 79, 80, 81, 82, 85, 87]


def assign_thirds_to_matches(
    best_8_thirds: List[Tuple[str, TableRow]]
) -> Dict[int, Tuple[str, TableRow]]:
    """
    Robust assignment of 8 third-place teams into the 8 FIFA third-place slots.

    Uses backtracking to avoid greedy dead-ends.
    Preference: when multiple assignments are possible, it tries higher-ranked
    third-place teams first (earlier in best_8_thirds).
    """
    slots = THIRD_MATCH_ORDER[:]  # [74, 77, 79, 80, 81, 82, 85, 87]

    # Keep thirds in rank order (best -> worst)
    thirds = best_8_thirds[:]

    # Precompute candidates for each slot by index into `thirds`
    def candidates_for_slot(slot: int, available_idxs: set[int]) -> List[int]:
        allowed = THIRD_SLOT_ALLOWED_GROUPS[slot]
        # preserve ranking order by scanning in increasing index
        return [i for i in range(len(thirds)) if i in available_idxs and thirds[i][0] in allowed]

    # Backtracking
    assignment: Dict[int, int] = {}  # slot -> index into thirds
    available = set(range(len(thirds)))

    def backtrack() -> bool:
        if len(assignment) == len(slots):
            return True

        # Choose the next slot using MRV heuristic (fewest options)
        best_slot = None
        best_cands: List[int] = []

        for slot in slots:
            if slot in assignment:
                continue
            cands = candidates_for_slot(slot, available)
            if not cands:
                return False
            if best_slot is None or len(cands) < len(best_cands):
                best_slot = slot
                best_cands = cands

        assert best_slot is not None

        # Try candidates in rank order (best third-place first)
        for idx in best_cands:
            assignment[best_slot] = idx
            available.remove(idx)

            if backtrack():
                return True

            # undo
            available.add(idx)
            del assignment[best_slot]

        return False

    ok = backtrack()
    if not ok:
        # Helpful debug message
        remaining_groups = [thirds[i][0] for i in sorted(available)]
        raise ValueError(
            "No feasible assignment of third-place teams to FIFA slots. "
            f"Remaining third groups at failure: {remaining_groups}"
        )

    # Convert slot->idx into slot->(group,row)
    return {slot: thirds[idx] for slot, idx in assignment.items()}


from typing import Set, Optional

def _team_from_qualifiers(q: List[Tuple[str, TableRow]], group: str) -> str:
    for g, row in q:
        if g == group:
            return row.team
    raise KeyError(f"Missing group {group} in qualifiers.")

def _assign_thirds_to_slots(
    best_thirds: List[Tuple[str, TableRow]],
    slot_allowed: Dict[str, Set[str]],
) -> Dict[str, str]:
    """
    Returns mapping slot_name -> group_letter for the 8 third-place slots,
    using only the groups that actually qualified as best_thirds.
    """
    third_groups = [g for g, _ in best_thirds]
    third_set = set(third_groups)

    # only groups that qualified matter; intersect allowed sets
    allowed = {slot: (slot_allowed[slot] & third_set) for slot in slot_allowed}

    # order slots by most constrained first (helps backtracking)
    slots = sorted(allowed.keys(), key=lambda s: len(allowed[s]))

    used: Set[str] = set()
    assignment: Dict[str, str] = {}

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        slot = slots[i]
        # try each possible group for this slot
        for g in sorted(allowed[slot]):
            if g in used:
                continue
            used.add(g)
            assignment[slot] = g
            if backtrack(i + 1):
                return True
            used.remove(g)
            del assignment[slot]
        return False

    if not backtrack(0):
        # helpful debug message
        msg = []
        msg.append(f"Qualified 3rd-place groups: {sorted(third_set)}")
        for slot in slots:
            msg.append(f"{slot} allows {sorted(slot_allowed[slot])}, viable={sorted(allowed[slot])}")
        raise ValueError("No valid FIFA third-place assignment found.\n" + "\n".join(msg))

    return assignment

def build_round_of_32_pairs_fifa(
    winners: List[Tuple[str, TableRow]],
    runners: List[Tuple[str, TableRow]],
    best_thirds: List[Tuple[str, TableRow]],
) -> List[Tuple[str, str]]:
    """
    FIFA-style Round of 32 using your bracket screenshot/table.

    Third-place slots (8 total) are constrained by allowed group letters.
    We solve the assignment so each qualified 3rd-place group is used once.
    """

    # --- Third-place slot constraints (from your screenshot/table) ---
    # Slot names are arbitrary (T1..T8) but must match the match templates below.
    slot_allowed = {
        "T1": set(list("ABCDF")),   # Winner Group E vs 3rd Group A/B/C/D/F
        "T2": set(list("CDFGH")),   # Winner Group I vs 3rd Group C/D/F/G/H
        "T3": set(list("BEFIJ")),   # Winner Group D vs 3rd Group B/E/F/I/J
        "T4": set(list("AEHIJ")),   # Winner Group G vs 3rd Group A/E/H/I/J
        "T5": set(list("CEFHI")),   # Winner Group A vs 3rd Group C/E/F/H/I
        "T6": set(list("EHIJK")),   # Winner Group L vs 3rd Group E/H/I/J/K
        "T7": set(list("EFGIJ")),   # Winner Group B vs 3rd Group E/F/G/I/J
        "T8": set(list("DEIJL")),   # Winner Group K vs 3rd Group D/E/I/J/L
    }

    third_slot_to_group = _assign_thirds_to_slots(best_thirds, slot_allowed)

    # helper to fetch the actual TEAM NAME for the third-place team from a group letter
    third_team_by_group = {g: row.team for g, row in best_thirds}

    def third_team(slot: str) -> str:
        g = third_slot_to_group[slot]
        return third_team_by_group[g]

    # --- Match templates in bracket order (your screenshot top-to-bottom) ---
    # Each entry returns (teamA, teamB).
    matches: List[Tuple[str, str]] = [
        (_team_from_qualifiers(winners, "E"), third_team("T1")),                 # W E vs 3rd ABCDF
        (_team_from_qualifiers(winners, "I"), third_team("T2")),                 # W I vs 3rd CDFGH
        (_team_from_qualifiers(runners, "A"), _team_from_qualifiers(runners, "B")),  # R A vs R B
        (_team_from_qualifiers(winners, "F"), _team_from_qualifiers(runners, "C")),  # W F vs R C
        (_team_from_qualifiers(runners, "K"), _team_from_qualifiers(runners, "L")),  # R K vs R L
        (_team_from_qualifiers(winners, "H"), _team_from_qualifiers(runners, "J")),  # W H vs R J
        (_team_from_qualifiers(winners, "D"), third_team("T3")),                 # W D vs 3rd BEFIJ
        (_team_from_qualifiers(winners, "G"), third_team("T4")),                 # W G vs 3rd AEHIJ
        (_team_from_qualifiers(winners, "C"), _team_from_qualifiers(runners, "F")),  # W C vs R F
        (_team_from_qualifiers(runners, "E"), _team_from_qualifiers(runners, "I")),  # R E vs R I
        (_team_from_qualifiers(winners, "A"), third_team("T5")),                 # W A vs 3rd CEFHI
        (_team_from_qualifiers(winners, "L"), third_team("T6")),                 # W L vs 3rd EHIJK
        (_team_from_qualifiers(winners, "J"), _team_from_qualifiers(runners, "H")),  # W J vs R H
        (_team_from_qualifiers(runners, "D"), _team_from_qualifiers(runners, "G")),  # R D vs R G
        (_team_from_qualifiers(winners, "B"), third_team("T7")),                 # W B vs 3rd EFGIJ
        (_team_from_qualifiers(winners, "K"), third_team("T8")),                 # W K vs 3rd DEIJL
    ]

    return matches

def build_round_of_16_pairs(
    winners: List[Tuple[str, TableRow]],
    best_runners: List[Tuple[str, TableRow]],
) -> List[Tuple[str, str]]:
    """
    For 11 groups + 5 runners-up = 16 teams:
      - Take TOP 8 group winners as "seeds"
      - The remaining 3 group winners + 5 runners-up are the 8 "unseeded" teams
      - Pair seed #1 vs unseeded #8, #2 vs #7, ... (deterministic bracket)
      - Avoid same-group pairing where possible by swapping within the unseeded list.

    Returns 8 pairs (seed_team, unseeded_team) in bracket order.
    """
    # Rank winners best->worst
    winners_sorted = sorted(winners, key=lambda gr: rank_row(gr[1]), reverse=True)

    if len(winners_sorted) < 8:
        raise ValueError(f"Need at least 8 group winners, got {len(winners_sorted)}")

    seeds = winners_sorted[:8]          # 8 best winners
    leftover_winners = winners_sorted[8:]  # remaining winners (should be 3)

    # Rank runners-up best->worst (we already passed "best 5", but keep deterministic ordering)
    runners_sorted = sorted(best_runners, key=lambda gr: rank_row(gr[1]), reverse=True)

    # Unseeded pool = leftover winners + best runners-up
    unseeded = leftover_winners + runners_sorted

    if len(unseeded) != 8:
        raise ValueError(f"Expected 8 unseeded teams (3 winners + 5 runners), got {len(unseeded)}")

    # Sort unseeded worst->best so top seed plays weakest
    unseeded_sorted = sorted(unseeded, key=lambda gr: rank_row(gr[1]), reverse=False)

    # Initial pairing: seed i vs unseeded i
    pairs = [(seeds[i][1].team, unseeded_sorted[i][1].team) for i in range(8)]

    # Try to avoid same-group matchups by swapping unseeded opponents
    # (only relevant if unseeded includes a runner-up from same group as a seed)
    for i in range(8):
        seed_group = seeds[i][0]
        opp_group = None

        # find opponent's group (search in unseeded list)
        for g, row in unseeded_sorted:
            if row.team == pairs[i][1]:
                opp_group = g
                break

        if opp_group == seed_group:
            # swap with a later unseeded opponent that doesn't conflict
            for j in range(i + 1, 8):
                opp_group_j = None
                for g, row in unseeded_sorted:
                    if row.team == pairs[j][1]:
                        opp_group_j = g
                        break
                if opp_group_j != seed_group:
                    # swap opponents
                    pairs[i] = (pairs[i][0], pairs[j][1])
                    pairs[j] = (pairs[j][0], pairs[i][1])
                    break

    return pairs

def knockout_winner_et_pens(a: str, b: str) -> str:
    # 90 minutes
    ga90, gb90 = play_match(a, b)
    if ga90 > gb90:
        return a
    if gb90 > ga90:
        return b

    # extra time (30 mins)
    lam_a, lam_b = expected_lambdas(a, b)
    ga_et = poisson_sample(lam_a / 3.0)
    gb_et = poisson_sample(lam_b / 3.0)

    ga = ga90 + ga_et
    gb = gb90 + gb_et

    if ga > gb:
        return a
    if gb > ga:
        return b

    # penalties
    return penalty_winner(a, b)

def run_fixed_bracket_32(ro32_pairs: List[Tuple[str, str]]) -> str:
    # Round of 32: 16 matches -> 16 winners
    r32_winners = [knockout_winner_et_pens(a, b) for a, b in ro32_pairs]
    if len(r32_winners) != 16:
        raise ValueError(f"Expected 16 R32 winners, got {len(r32_winners)}")

    # Round of 16: 8 matches -> 8 winners
    r16_pairs = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_winners = [knockout_winner_et_pens(a, b) for a, b in r16_pairs]
    if len(r16_winners) != 8:
        raise ValueError(f"Expected 8 R16 winners, got {len(r16_winners)}")

    # Quarterfinals: 4 matches -> 4 winners
    qf_pairs = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_winners = [knockout_winner_et_pens(a, b) for a, b in qf_pairs]

    # Semifinals: 2 matches -> 2 winners
    sf_pairs = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, 4, 2)]
    sf_winners = [knockout_winner_et_pens(a, b) for a, b in sf_pairs]

    # Final: 1 match -> champion
    return knockout_winner_et_pens(sf_winners[0], sf_winners[1])


def simulate_group_verbose(group_name: str, teams: List[str]) -> Tuple[List[TableRow], List[str]]:
    rows = {t: TableRow(team=t) for t in teams}
    logs: List[str] = []

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]

            lam_a, lam_b = expected_lambdas(a, b)   # expected goals (not rounded)
            ga, gb = play_match(a, b)               # one stochastic sample

            if ga > gb:
                outcome = f"{a} win"
            elif gb > ga:
                outcome = f"{b} win"
            else:
                outcome = "draw"

            logs.append(
                f"[Group {group_name}] {a} {ga}-{gb} {b}  "
                f"(xG {lam_a:.2f}-{lam_b:.2f}) -> {outcome}"
            )

            rows[a].gf += ga
            rows[a].ga += gb
            rows[b].gf += gb
            rows[b].ga += ga

            if ga > gb:
                rows[a].pts += 3
            elif gb > ga:
                rows[b].pts += 3
            else:
                rows[a].pts += 1
                rows[b].pts += 1

    table = list(rows.values())
    table.sort(key=lambda r: (r.pts, r.gd, r.gf, r.team), reverse=True)
    return table, logs


def play_knockout_match_verbose(a: str, b: str) -> Tuple[str, str]:
    ga90, gb90 = play_match(a, b)
    if ga90 > gb90:
        return a, f"{a} {ga90}-{gb90} {b} -> {a}"
    if gb90 > ga90:
        return b, f"{a} {ga90}-{gb90} {b} -> {b}"

    lam_a, lam_b = expected_lambdas(a, b)
    ga_et = poisson_sample(lam_a / 3.0)
    gb_et = poisson_sample(lam_b / 3.0)

    ga = ga90 + ga_et
    gb = gb90 + gb_et

    if ga > gb:
        return a, f"{a} {ga}-{gb} {b} -> {a} (ET)"
    if gb > ga:
        return b, f"{a} {ga}-{gb} {b} -> {b} (ET)"

    winner = penalty_winner(a, b)
    return winner, f"{a} {ga}-{gb} {b} -> {winner} (pens)"


def print_group_table(group_name: str, table: List[TableRow]) -> None:
    print(f"\nGroup {group_name} table:")
    for r in table:
        print(f"  {r.team:15s} pts={r.pts:2d} gf={r.gf:2d} ga={r.ga:2d} gd={r.gd:3d}")


def simulate_one_world_cup_verbose() -> str:
    print("\n==============================")
    print(" ONE SIMULATED WORLD CUP RUN")
    print("==============================\n")

    group_tables: Dict[str, List[TableRow]] = {}

    print("=== GROUP STAGE MATCHES ===")
    for g, teams in GROUPS.items():
        table, logs = simulate_group_verbose(g, teams)
        group_tables[g] = table
        for line in logs:
            print(line)

    print("\n=== GROUP TABLES ===")
    for g in GROUPS.keys():
        print_group_table(g, group_tables[g])

    winners, runners, thirds = select_qualifiers(group_tables)
    best_8_thirds = pick_best_third_place(thirds, k=8)
    ro32_pairs = build_round_of_32_pairs_fifa(winners, runners, best_8_thirds)

    print("\n=== QUALIFIERS ===")
    print("Group winners:", [row.team for _, row in winners])
    print("Group runners-up:", [row.team for _, row in runners])
    print("Best 8 third-place:", [row.team for _, row in best_8_thirds])

    ro32_pairs = build_round_of_32_pairs_fifa(winners, runners, best_8_thirds)

    print("\n=== ROUND OF 32 ===")
    for i, (a, b) in enumerate(ro32_pairs, start=1):
        print(f"  Match {i}: {a} vs {b}")

# then run knockouts in stages like you already do, but starting from ro32_pairs

    print("\n=== KNOCKOUT MATCHES ===")

    # Round of 32
    r32_winners: List[str] = []
    print("\n-- Round of 32 --")
    for a, b in ro32_pairs:
        w, line = play_knockout_match_verbose(a, b)
        print(" ", line)
        r32_winners.append(w)

    # Round of 16
    print("\n-- Round of 16 --")
    r16_pairs = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_winners: List[str] = []
    for a, b in r16_pairs:
        w, line = play_knockout_match_verbose(a, b)
        print(" ", line)
        r16_winners.append(w)

    # Quarterfinals
    print("\n-- Quarterfinals --")
    qf_pairs = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_winners: List[str] = []
    for a, b in qf_pairs:
        w, line = play_knockout_match_verbose(a, b)
        print(" ", line)
        qf_winners.append(w)

    # Semifinals
    print("\n-- Semifinals --")
    sf_pairs = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, 4, 2)]
    sf_winners: List[str] = []
    for a, b in sf_pairs:
        w, line = play_knockout_match_verbose(a, b)
        print(" ", line)
        sf_winners.append(w)

    # Final
    print("\n-- Final --")
    champ, line = play_knockout_match_verbose(sf_winners[0], sf_winners[1])
    print(" ", line)
    print(f"\n🏆 Champion of this run: {champ}\n")  

    return champ

# =========================
# 6) FULL TOURNAMENT SIM (NON-RANDOM BRACKET)
# =========================

def simulate_tournament_once() -> str:
    group_tables: Dict[str, List[TableRow]] = {}

    for g, teams in GROUPS.items():
        group_tables[g] = simulate_group(teams)

    winners, runners, thirds = select_qualifiers(group_tables)
    best_8_thirds = pick_best_third_place(thirds, k=8)
    ro32_pairs = build_round_of_32_pairs_fifa(winners, runners, best_8_thirds)
    champ = run_fixed_bracket_32(ro32_pairs)
    return champ

def simulate_group_verbose_return(group_name: str, teams: List[str]) -> Tuple[List[TableRow], List[Tuple[str, str, int, int]]]:
    """
    Returns:
      - final group table
      - list of matches as tuples: (teamA, teamB, goalsA, goalsB)
    """
    rows = {t: TableRow(team=t) for t in teams}
    matches: List[Tuple[str, str, int, int]] = []

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = play_match(a, b)
            matches.append((a, b, ga, gb))

            rows[a].gf += ga
            rows[a].ga += gb
            rows[b].gf += gb
            rows[b].ga += ga

            if ga > gb:
                rows[a].pts += 3
            elif gb > ga:
                rows[b].pts += 3
            else:
                rows[a].pts += 1
                rows[b].pts += 1

    table = list(rows.values())
    table.sort(key=lambda r: (r.pts, r.gd, r.gf, r.team), reverse=True)
    return table, matches


# =========================
# MAIN
# =========================

def main() -> None:
    print("Starting wc_model.py ...", flush=True)

    print("Loading cleaned CSVs...", flush=True)
    _all_data = load_all_csvs_from_folder(CLEAN_DIR)
    print(f"Loaded {len(_all_data):,} rows.", flush=True)

    print("Building strengths using Poisson GLM...", flush=True)
    global strengths, BASE_GOALS, USED_FEATURES
    strengths, BASE_GOALS, USED_FEATURES = build_strengths_from_all_columns(_all_data)
    print("Finished fitting strengths.\n", flush=True)

    all_teams = sorted({t for teams in GROUPS.values() for t in teams})
    missing = [t for t in all_teams if t not in strengths.index]

    print(f"Features used (all numeric unless collinear): {len(USED_FEATURES)}")
    print(f"BASE_GOALS (learned from data): {BASE_GOALS:.3f}")
    print(f"Teams in groups: {len(all_teams)}")
    print(f"Teams missing from strengths (fallback to avg 1.0): {missing}\n", flush=True)

    # ---- PRINT ONE FULL TOURNAMENT ----
    print("Simulating one full World Cup with all matches printed...\n", flush=True)
    simulate_one_world_cup_verbose()

    # ---- MONTE CARLO FOR PROBABILITIES ----
    print("\nRunning Monte Carlo simulations...", flush=True)
    counts = {t: 0 for t in all_teams}

    for i in range(N_SIMS):
        champ = simulate_tournament_once()
        counts[champ] = counts.get(champ, 0) + 1

        # progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Completed {i+1} / {N_SIMS} sims...", flush=True)

    results = sorted(counts.items(), key=lambda x: -x[1])

    print("\n=== Champion probabilities ===")
    for team, c in results[:20]:
        if c > 0:
            print(f"{team:20s}  {c / N_SIMS:.3%}")


if __name__ == "__main__":
    main()
