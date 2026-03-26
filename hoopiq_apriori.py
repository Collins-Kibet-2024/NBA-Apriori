"""
HoopIQ — Play Sequence Pattern Mining (Apriori)
================================================
Drop your NBA play-by-play CSV into the same folder and run:

    python hoopiq_apriori.py --file your_pbp_data.csv

Or pipe a game_id directly if you have nba_api set up:

    python hoopiq_apriori.py --live --team BOS --season 2023-24 --games 20

CSV expected columns (nba_api PlayByPlayV2 schema):
    GAME_ID, EVENTNUM, PERIOD, PCTIMESTRING,
    EVENTMSGTYPE, EVENTMSGACTIONTYPE,
    HOMEDESCRIPTION, VISITORDESCRIPTION,
    PLAYER1_ID, PLAYER2_ID, SCORE, SCOREMARGIN

Outputs:
  - Console: benchmark tables, top rules, event distribution
  - hoopiq_results.csv : full association rule table
  - hoopiq_patterns.json: structured JSON for web app consumption
"""

import argparse
import json
import time
import warnings
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pip
pip.main(['install', 'nba_api', 'mlxtend', 'tabulate', 'pandas', 'numpy'])
import numpy as np
import pandas as pd
import pip
import requests
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# EVENT TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

# Maps (EVENTMSGTYPE, EVENTMSGACTIONTYPE) → human-readable action label
EVENT_MAP = {
    # ── Field Goals Made ──────────────────────────────────────────────────────
    (1, 1):   "FGM_JUMP",       (1, 2):   "FGM_JUMP",
    (1, 3):   "FGM_JUMP",       (1, 4):   "FGM_JUMP",
    (1, 5):   "FGM_LAYUP",      (1, 6):   "FGM_LAYUP",
    (1, 7):   "FGM_DUNK",       (1, 8):   "FGM_DUNK",
    (1, 9):   "FGM_DUNK",       (1, 41):  "FGM_DRIVE",
    (1, 43):  "FGM_ALLEY",      (1, 44):  "FGM_ALLEY",
    (1, 47):  "FGM_HOOK",       (1, 52):  "FGM_PUTBACK",
    (1, 57):  "FGM_HOOK",       (1, 58):  "FGM_TURNAROUND",
    (1, 66):  "FGM_JUMP",       (1, 67):  "FGM_JUMP",
    (1, 72):  "FGM_PULLUP",     (1, 73):  "FGM_STEPBACK",
    (1, 75):  "FGM_FLOATER",    (1, 79):  "FGM_FADEAWAY",
    (1, 93):  "FGM_LAYUP",      (1, 97):  "FGM_TURNAROUND",
    (1, 98):  "FGM_3PT",        (1, 99):  "FGM_3PT",
    (1, 100): "FGM_3PT",        (1, 101): "FGM_3PT",
    (1, 102): "FGM_3PT",        (1, 103): "FGM_3PT",
    (1, 104): "FGM_3PT",        (1, 105): "FGM_3PT",
    (1, 106): "FGM_3PT",        (1, 107): "FGM_3PT",
    (1, 108): "FGM_PULLUP",     (1, 109): "FGM_PULLUP",

    # ── Field Goals Missed ────────────────────────────────────────────────────
    (2, 1):   "FGA_JUMP",       (2, 2):   "FGA_JUMP",
    (2, 3):   "FGA_JUMP",       (2, 5):   "FGA_LAYUP",
    (2, 6):   "FGA_LAYUP",      (2, 7):   "FGA_DUNK",
    (2, 41):  "FGA_DRIVE",      (2, 47):  "FGA_HOOK",
    (2, 58):  "FGA_TURNAROUND", (2, 66):  "FGA_JUMP",
    (2, 72):  "FGA_PULLUP",     (2, 73):  "FGA_STEPBACK",
    (2, 75):  "FGA_FLOATER",    (2, 79):  "FGA_FADEAWAY",
    (2, 98):  "FGA_3PT",        (2, 99):  "FGA_3PT",
    (2, 100): "FGA_3PT",        (2, 101): "FGA_3PT",
    (2, 102): "FGA_3PT",        (2, 108): "FGA_PULLUP",

    # ── Free Throws ───────────────────────────────────────────────────────────
    (3, 10):  "FT_MADE",        (3, 11):  "FT_MADE",
    (3, 12):  "FT_MADE",        (3, 13):  "FT_MISSED",
    (3, 14):  "FT_MISSED",      (3, 15):  "FT_MISSED",
    (3, 16):  "FT_MADE",        (3, 17):  "FT_MADE",
    (3, 18):  "FT_MADE",        (3, 19):  "FT_MISSED",
    (3, 20):  "FT_MISSED",      (3, 21):  "FT_MISSED",

    # ── Rebounds ──────────────────────────────────────────────────────────────
    (4, 0):   "REB_TEAM",       (4, 1):   "REB_OFF",
    (4, 2):   "REB_DEF",

    # ── Turnovers ─────────────────────────────────────────────────────────────
    (5, 1):   "TOV_BALL",       (5, 2):   "TOV_TRAVEL",
    (5, 3):   "TOV_DOUBLE",     (5, 4):   "TOV_BALL",
    (5, 8):   "TOV_BALL",       (5, 11):  "TOV_STEAL",
    (5, 15):  "TOV_BALL",       (5, 17):  "TOV_SHOT_CLOCK",
    (5, 37):  "TOV_OUT_BOUNDS", (5, 38):  "TOV_BALL",
    (5, 39):  "TOV_LANE",       (5, 40):  "TOV_LANE",
    (5, 45):  "TOV_BALL",

    # ── Fouls ─────────────────────────────────────────────────────────────────
    (6, 1):   "FOUL_PERSONAL",  (6, 2):   "FOUL_SHOOTING",
    (6, 3):   "FOUL_CHARGE",    (6, 4):   "FOUL_INDIR",
    (6, 5):   "FOUL_FLAG1",     (6, 6):   "FOUL_FLAG2",
    (6, 9):   "FOUL_PERSONAL",  (6, 14):  "FOUL_OFF",
    (6, 15):  "FOUL_PERSONAL",  (6, 17):  "FOUL_SHOOTING",
    (6, 18):  "FOUL_SHOOTING",  (6, 26):  "FOUL_PERSONAL",
    (6, 27):  "FOUL_CHARGE",    (6, 28):  "FOUL_PERSONAL",

    # ── Violations ────────────────────────────────────────────────────────────
    (7, 1):   "VIO_LANE",       (7, 2):   "VIO_GOALTEND",
    (7, 3):   "VIO_JUMP",       (7, 4):   "VIO_KICK",

    # ── Jump ball / misc ──────────────────────────────────────────────────────
    (10, 0):  "JUMP_BALL",
}

SCORING_EVENTS = frozenset({
    "FGM_JUMP", "FGM_LAYUP", "FGM_DUNK", "FGM_DRIVE",
    "FGM_ALLEY", "FGM_PULLUP", "FGM_STEPBACK", "FGM_FLOATER",
    "FGM_FADEAWAY", "FGM_TURNAROUND", "FGM_HOOK", "FGM_PUTBACK",
    "FGM_3PT", "FT_MADE",
})

TURNOVER_EVENTS = frozenset({
    "TOV_STEAL", "TOV_BALL", "TOV_TRAVEL", "TOV_DOUBLE",
    "TOV_SHOT_CLOCK", "TOV_OUT_BOUNDS", "TOV_LANE",
})

TERMINAL_EVENTS = SCORING_EVENTS | TURNOVER_EVENTS

# Filtered out mid-possession (don't add signal to sequence)
NOISE_EVENTS = frozenset({
    "REB_DEF", "REB_TEAM", "FOUL_PERSONAL", "FOUL_SHOOTING",
    "FOUL_CHARGE", "FOUL_INDIR", "FOUL_FLAG1", "FOUL_FLAG2",
    "FOUL_OFF", "FT_MISSED", "VIO_LANE", "VIO_GOALTEND",
    "VIO_JUMP", "VIO_KICK", "JUMP_BALL",
})

# Friendly display names for the report
ACTION_LABELS = {
    "FGM_JUMP":       "Jump Shot (Made)",
    "FGM_LAYUP":      "Layup (Made)",
    "FGM_DUNK":       "Dunk (Made)",
    "FGM_DRIVE":      "Driving Layup (Made)",
    "FGM_ALLEY":      "Alley-Oop (Made)",
    "FGM_PULLUP":     "Pull-Up Jumper (Made)",
    "FGM_STEPBACK":   "Step-Back (Made)",
    "FGM_FLOATER":    "Floater (Made)",
    "FGM_FADEAWAY":   "Fadeaway (Made)",
    "FGM_TURNAROUND": "Turnaround (Made)",
    "FGM_HOOK":       "Hook Shot (Made)",
    "FGM_PUTBACK":    "Putback (Made)",
    "FGM_3PT":        "3-Pointer (Made)",
    "FT_MADE":        "Free Throw (Made)",
    "FGA_JUMP":       "Jump Shot (Missed)",
    "FGA_LAYUP":      "Layup (Missed)",
    "FGA_DUNK":       "Dunk (Missed)",
    "FGA_DRIVE":      "Drive (Missed)",
    "FGA_3PT":        "3-Pointer (Missed)",
    "FGA_PULLUP":     "Pull-Up (Missed)",
    "FGA_FLOATER":    "Floater (Missed)",
    "FGA_FADEAWAY":   "Fadeaway (Missed)",
    "FGA_TURNAROUND": "Turnaround (Missed)",
    "FGA_HOOK":       "Hook Shot (Missed)",
    "FGA_STEPBACK":   "Step-Back (Missed)",
    "REB_OFF":        "Offensive Rebound",
    "REB_DEF":        "Defensive Rebound",
    "TOV_STEAL":      "Stolen — Turnover",
    "TOV_BALL":       "Ball-Handling Turnover",
    "TOV_TRAVEL":     "Travel",
    "TOV_SHOT_CLOCK": "Shot Clock Violation",
    "TOV_OUT_BOUNDS": "Out of Bounds — Turnover",
    "TOV_LANE":       "Lane Violation",
}


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_event(row) -> str:
    key = (int(row["EVENTMSGTYPE"]), int(row["EVENTMSGACTIONTYPE"]))
    if key in EVENT_MAP:
        return EVENT_MAP[key]
    # Fallback by top-level type
    fallbacks = {
        1: "FGM_OTHER", 2: "FGA_OTHER", 3: "FT_OTHER",
        4: "REBOUND",   5: "TURNOVER",  6: "FOUL",
        7: "VIOLATION",
    }
    return fallbacks.get(int(row["EVENTMSGTYPE"]), f"EVT_{int(row['EVENTMSGTYPE'])}")


def normalize_live_action(action: dict) -> tuple[str, int]:
    action_type = str(action.get("actionType", "")).lower()
    sub_type = str(action.get("subType", "")).lower()
    descriptor = str(action.get("descriptor", "")).lower()
    shot_result = str(action.get("shotResult", "")).lower()
    qualifiers = {str(q).lower() for q in action.get("qualifiers", [])}

    if action_type == "period":
        return "PERIOD_MARKER", 12

    if action_type == "substitution":
        return "SUBSTITUTION", 8

    if action_type == "jumpball":
        return "JUMP_BALL", 10

    if action_type == "rebound":
        return ("REB_OFF", 4) if sub_type == "offensive" else ("REB_DEF", 4)

    if action_type == "freethrow":
        return ("FT_MADE", 3) if shot_result == "made" else ("FT_MISSED", 3)

    if action_type == "turnover":
        if "shot clock" in sub_type:
            return "TOV_SHOT_CLOCK", 5
        if "out-of-bounds" in sub_type:
            return "TOV_OUT_BOUNDS", 5
        if "travel" in sub_type:
            return "TOV_TRAVEL", 5
        if "double" in sub_type:
            return "TOV_DOUBLE", 5
        return "TOV_BALL", 5

    if action_type == "steal":
        return "TOV_STEAL", 5

    if action_type == "foul":
        if sub_type == "offensive":
            return "FOUL_OFF", 6
        if descriptor == "shooting":
            return "FOUL_SHOOTING", 6
        if sub_type == "charge":
            return "FOUL_CHARGE", 6
        return "FOUL_PERSONAL", 6

    if action_type == "2pt":
        shot_prefix = "FGM" if shot_result == "made" else "FGA"
        if sub_type == "layup":
            if descriptor == "driving":
                return f"{shot_prefix}_DRIVE", 1 if shot_prefix == "FGM" else 2
            return f"{shot_prefix}_LAYUP", 1 if shot_prefix == "FGM" else 2
        if sub_type == "dunk":
            return f"{shot_prefix}_DUNK", 1 if shot_prefix == "FGM" else 2
        if sub_type == "hook":
            return f"{shot_prefix}_HOOK", 1 if shot_prefix == "FGM" else 2
        if descriptor == "pullup":
            return f"{shot_prefix}_PULLUP", 1 if shot_prefix == "FGM" else 2
        if descriptor == "turnaround":
            return f"{shot_prefix}_TURNAROUND", 1 if shot_prefix == "FGM" else 2
        if descriptor == "fadeaway":
            return f"{shot_prefix}_FADEAWAY", 1 if shot_prefix == "FGM" else 2
        if descriptor == "step back":
            return f"{shot_prefix}_STEPBACK", 1 if shot_prefix == "FGM" else 2
        if descriptor == "floating":
            return f"{shot_prefix}_FLOATER", 1 if shot_prefix == "FGM" else 2
        return f"{shot_prefix}_JUMP", 1 if shot_prefix == "FGM" else 2

    if action_type == "3pt":
        return ("FGM_3PT", 1) if shot_result == "made" else ("FGA_3PT", 2)

    if action_type in {"timeout", "block", "game"}:
        return "NOISE", 0

    return f"LIVE_{action_type.upper() or 'UNKNOWN'}", 0


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_sequences(pbp_df: pd.DataFrame,
                      min_len: int = 2,
                      max_len: int = 5) -> tuple[list, list]:
    """
    Segment play-by-play into offensive possession sequences.

    A possession ends when:
      - A scoring event occurs          (sequence tagged as scoring)
      - A turnover event occurs         (sequence tagged as non-scoring)
      - A period marker is hit          (flush and reset)

    Returns:
        all_seqs     : list[list[str]]  — every possession sequence
        scoring_seqs : list[list[str]]  — only possessions ending in a score
    """
    pbp = pbp_df.copy()
    pbp["EVENTMSGTYPE"] = pd.to_numeric(pbp["EVENTMSGTYPE"], errors="coerce").fillna(0).astype(int)

    if "ACTION" in pbp.columns:
        pbp["ACTION"] = pbp["ACTION"].astype(str)
    else:
        pbp["EVENTMSGACTIONTYPE"] = pd.to_numeric(pbp["EVENTMSGACTIONTYPE"], errors="coerce").fillna(0).astype(int)
        pbp["ACTION"] = pbp.apply(normalize_event, axis=1)

    all_seqs:     list[list[str]] = []
    scoring_seqs: list[list[str]] = []

    for game_id, gdf in pbp.groupby("GAME_ID"):
        gdf     = gdf.sort_values("EVENTNUM").reset_index(drop=True)
        current: list[str] = []

        for _, row in gdf.iterrows():
            action = row["ACTION"]
            etype  = row["EVENTMSGTYPE"]

            # Period start/end and substitutions → flush possession
            if etype in (8, 9, 12, 13):
                if len(current) >= min_len:
                    seq = current[:max_len]
                    all_seqs.append(seq)
                    if seq[-1] in SCORING_EVENTS:
                        scoring_seqs.append(seq)
                current = []
                continue

            # Noise events don't add sequence signal (skip mid-possession)
            if action in NOISE_EVENTS and current:
                continue

            current.append(action)

            # Possession ends
            if action in TERMINAL_EVENTS:
                seq = current[:max_len]
                if len(seq) >= min_len:
                    all_seqs.append(seq)
                    if action in SCORING_EVENTS:
                        scoring_seqs.append(seq)
                current = []

        # Flush any trailing sequence at game end
        if len(current) >= min_len:
            all_seqs.append(current[:max_len])

    return all_seqs, scoring_seqs


# ══════════════════════════════════════════════════════════════════════════════
# APRIORI PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def encode_transactions(sequences: list[list[str]]) -> pd.DataFrame:
    """One-hot encode sequences into an item-transaction matrix."""
    te       = TransactionEncoder()
    te_array = te.fit(sequences).transform(sequences)
    return pd.DataFrame(te_array, columns=te.columns_)


def run_apriori_pipeline(all_seqs:      list[list[str]],
                         scoring_seqs:  list[list[str]],
                         min_support:   float = 0.05,
                         min_confidence:float = 0.30,
                         min_lift:      float = 1.0,
                         max_len:       int   = 5) -> tuple[pd.DataFrame, dict]:
    """
    Full Apriori pipeline:
      1. Encode all sequences as transaction matrix
      2. Mine frequent itemsets
      3. Generate association rules
      4. Filter to rules whose consequent is a scoring event
      5. Return sorted DataFrame + performance metrics
    """
    metrics: dict = {}
    t0 = time.perf_counter()

    # ── Encode ────────────────────────────────────────────────────────────────
    basket_df  = encode_transactions(all_seqs)
    metrics["encode_time"] = time.perf_counter() - t0

    # ── Frequent itemsets ─────────────────────────────────────────────────────
    t1 = time.perf_counter()
    freq_items = apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    metrics["apriori_time"]   = time.perf_counter() - t1
    metrics["n_freq_itemsets"]= len(freq_items)

    if freq_items.empty:
        return pd.DataFrame(), metrics

    # ── Association rules ─────────────────────────────────────────────────────
    t2 = time.perf_counter()
    rules = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(freq_items),
    )
    metrics["rules_time"] = time.perf_counter() - t2
    metrics["n_rules_raw"]= len(rules)

    if rules.empty:
        return pd.DataFrame(), metrics

    # ── Filter to scoring consequents ─────────────────────────────────────────
    rules["consequent_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules["antecedent_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))

    scoring_rules = rules[
        rules["consequents"].apply(lambda c: bool(c & SCORING_EVENTS))
    ].copy()

    scoring_rules = scoring_rules[scoring_rules["lift"] >= min_lift]
    scoring_rules = scoring_rules.sort_values("lift", ascending=False).reset_index(drop=True)

    metrics["n_scoring_rules"] = len(scoring_rules)
    metrics["total_time"]      = time.perf_counter() - t0

    return scoring_rules, metrics


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SWEEP BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def sweep_thresholds(all_seqs, scoring_seqs,
                     thresholds: list[float] = None) -> pd.DataFrame:
    """Run Apriori across multiple support thresholds and return benchmark table."""
    if thresholds is None:
        thresholds = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

    rows = []
    for thresh in thresholds:
        rules_df, m = run_apriori_pipeline(
            all_seqs, scoring_seqs,
            min_support=thresh, min_confidence=0.25, min_lift=1.0,
        )
        rows.append({
            "support":       f"{thresh:.0%}",
            "freq_itemsets": m.get("n_freq_itemsets", 0),
            "rules_raw":     m.get("n_rules_raw", 0),
            "scoring_rules": m.get("n_scoring_rules", 0),
            "encode_s":      round(m.get("encode_time", 0), 4),
            "apriori_s":     round(m.get("apriori_time", 0), 4),
            "rules_s":       round(m.get("rules_time", 0), 4),
            "total_s":       round(m.get("total_time", 0), 4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(title: str, width: int = 68):
    print(f"\n{'═'*width}\n  {title}\n{'═'*width}")


def print_dataset_summary(pbp_df, all_seqs, scoring_seqs):
    banner("DATASET SUMMARY")
    flat = list(itertools.chain.from_iterable(all_seqs))
    print(tabulate([
        ["Games",                     pbp_df["GAME_ID"].nunique()],
        ["Raw play events",           f"{len(pbp_df):,}"],
        ["Total sequences extracted", f"{len(all_seqs):,}"],
        ["Scoring sequences",         f"{len(scoring_seqs):,}"],
        ["Non-scoring sequences",     f"{len(all_seqs)-len(scoring_seqs):,}"],
        ["Scoring rate",              f"{len(scoring_seqs)/len(all_seqs)*100:.1f}%"],
        ["Avg sequence length",       f"{sum(len(s) for s in all_seqs)/len(all_seqs):.2f}"],
        ["Total events in sequences", f"{len(flat):,}"],
        ["Unique action types",       len({e for s in all_seqs for e in s})],
    ], tablefmt="rounded_outline"))


def print_benchmark(sweep_df: pd.DataFrame):
    banner("APRIORI BENCHMARK — Support Threshold Sweep")
    print(tabulate(
        sweep_df.values.tolist(),
        headers=["Support", "Freq Items", "Raw Rules", "Scoring Rules",
                 "Encode(s)", "Apriori(s)", "Rules(s)", "Total(s)"],
        tablefmt="rounded_outline",
    ))


def print_top_rules(rules_df: pd.DataFrame, n: int = 20):
    banner(f"TOP {n} SCORING ASSOCIATION RULES  (ranked by Lift)")
    if rules_df.empty:
        print("  No rules found. Try lowering --min-support or --min-confidence.")
        return

    rows = []
    for _, r in rules_df.head(n).iterrows():
        # Map to friendly label where available
        ant = " + ".join(
            ACTION_LABELS.get(a, a)
            for a in sorted(r["antecedents"])
        )
        con = ", ".join(
            ACTION_LABELS.get(c, c)
            for c in sorted(r["consequents"])
        )
        rows.append([ant, con,
                     f"{r['support']:.3f}",
                     f"{r['confidence']:.3f}",
                     f"{r['lift']:.2f}",
                     f"{int(r.get('antecedent support', 0) * len(rules_df)):,}" if False else "—"])

    print(tabulate(
        rows,
        headers=["Antecedent(s)", "Consequent (Score)", "Support", "Conf", "Lift", ""],
        tablefmt="rounded_outline",
        maxcolwidths=[42, 26, 8, 6, 6, 1],
    ))

    print(f"\n  Lift > 1.0 means the antecedent makes scoring MORE likely than chance.")
    print(f"  Confidence = P(score | sequence) — how often this pattern ends in a basket.")


def print_event_distribution(all_seqs: list):
    banner("EVENT FREQUENCY DISTRIBUTION")
    flat = list(itertools.chain.from_iterable(all_seqs))
    tc   = defaultdict(int)
    for ev in flat:
        tc[ev] += 1
    rows = sorted(tc.items(), key=lambda x: -x[1])
    print(tabulate(
        [[ACTION_LABELS.get(ev, ev), ev, cnt, f"{cnt/len(flat)*100:.1f}%"]
         for ev, cnt in rows],
        headers=["Action (friendly)", "Code", "Count", "% of Events"],
        tablefmt="rounded_outline",
    ))


def print_scoring_breakdown(scoring_seqs: list):
    banner("SCORING EVENT BREAKDOWN")
    terminal_counts = defaultdict(int)
    for seq in scoring_seqs:
        terminal_counts[seq[-1]] += 1
    total = sum(terminal_counts.values())
    rows  = sorted(terminal_counts.items(), key=lambda x: -x[1])
    print(tabulate(
        [[ACTION_LABELS.get(ev, ev), cnt, f"{cnt/total*100:.1f}%"]
         for ev, cnt in rows],
        headers=["Scoring Type", "Count", "% of Scores"],
        tablefmt="rounded_outline",
    ))


def print_top_sequences(all_seqs: list, scoring_seqs: list, n: int = 15):
    banner(f"TOP {n} RAW SEQUENCES (by frequency)")
    from collections import Counter
    seq_counter = Counter(tuple(s) for s in all_seqs)
    score_set   = set(tuple(s) for s in scoring_seqs)
    top         = seq_counter.most_common(n)
    rows = []
    for seq_tuple, count in top:
        is_score = "✓" if seq_tuple in score_set else "—"
        rows.append([
            " → ".join(seq_tuple),
            len(seq_tuple),
            count,
            f"{count/len(all_seqs)*100:.1f}%",
            is_score,
        ])
    print(tabulate(rows,
                   headers=["Sequence", "Len", "Count", "Support", "Scores?"],
                   tablefmt="rounded_outline",
                   maxcolwidths=[52, 4, 6, 8, 7]))


# ══════════════════════════════════════════════════════════════════════════════
# FILE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_results(rules_df: pd.DataFrame,
                   all_seqs: list,
                   scoring_seqs: list,
                   output_dir: str = "."):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── CSV: full rules table ──────────────────────────────────────────────────
    if not rules_df.empty:
        csv_path = out / "hoopiq_results.csv"
        export_df = rules_df[["antecedent_str", "consequent_str",
                               "support", "confidence", "lift"]].copy()
        export_df.columns = ["antecedents", "consequent", "support", "confidence", "lift"]
        export_df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Rules exported → {csv_path}")

    # ── JSON: structured for web app consumption ───────────────────────────────
    from collections import Counter
    seq_counter = Counter(tuple(s) for s in all_seqs)
    terminal_counts = defaultdict(int)
    for seq in scoring_seqs:
        terminal_counts[seq[-1]] += 1

    payload = {
        "meta": {
            "total_sequences":   len(all_seqs),
            "scoring_sequences": len(scoring_seqs),
            "scoring_rate":      round(len(scoring_seqs) / len(all_seqs), 4) if all_seqs else 0,
        },
        "top_sequences": [
            {"sequence": list(seq), "count": cnt,
             "support": round(cnt / len(all_seqs), 4)}
            for seq, cnt in seq_counter.most_common(50)
        ],
        "scoring_breakdown": {
            ACTION_LABELS.get(k, k): v
            for k, v in sorted(terminal_counts.items(), key=lambda x: -x[1])
        },
        "association_rules": [
            {
                "antecedents": sorted(r["antecedents"]),
                "consequent":  sorted(r["consequents"])[0],
                "support":     round(r["support"], 4),
                "confidence":  round(r["confidence"], 4),
                "lift":        round(r["lift"], 4),
            }
            for _, r in rules_df.iterrows()
        ] if not rules_df.empty else [],
    }

    json_path = out / "hoopiq_patterns.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ JSON payload exported → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE NBA API FETCH (optional — requires internet + nba_api)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_live(team_abbr: str, season: str, n_games: int) -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError:
        raise SystemExit("nba_api not installed. Run: pip install nba_api")

    print(f"  Querying LeagueGameLog for {team_abbr} ({season}) ...", end=" ", flush=True)
    log       = leaguegamelog.LeagueGameLog(season=season, timeout=30).get_data_frames()[0]
    game_ids  = log[log["TEAM_ABBREVIATION"] == team_abbr]["GAME_ID"].tolist()[:n_games]
    print(f"found {len(game_ids)} games")

    frames = []
    for i, gid in enumerate(game_ids, 1):
        print(f"  Fetching game {i}/{len(game_ids)}: {gid} ...", end=" ", flush=True)
        try:
            response = requests.get(
                f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json",
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            actions = payload.get("game", {}).get("actions", [])

            rows = []
            for action in actions:
                normalized_action, event_type = normalize_live_action(action)
                if normalized_action in {"NOISE", "TOV_STEAL"}:
                    continue

                rows.append({
                    "GAME_ID": gid,
                    "EVENTNUM": action.get("orderNumber", action.get("actionNumber", 0)),
                    "PERIOD": action.get("period"),
                    "PCTIMESTRING": action.get("clock"),
                    "EVENTMSGTYPE": event_type,
                    "EVENTMSGACTIONTYPE": 0,
                    "HOMEDESCRIPTION": action.get("description"),
                    "VISITORDESCRIPTION": action.get("description"),
                    "PLAYER1_ID": action.get("personId", 0),
                    "PLAYER2_ID": 0,
                    "SCORE": f"{action.get('scoreAway', '')}-{action.get('scoreHome', '')}",
                    "SCOREMARGIN": None,
                    "ACTION": normalized_action,
                })

            df = pd.DataFrame(rows)
            if df.empty:
                raise ValueError("empty play-by-play payload")
            frames.append(df)
            print(f"{len(df)} events")
        except Exception as e:
            print(f"SKIP ({e})")
        time.sleep(0.65)   # NBA API rate limit — do not remove

    if not frames:
        raise SystemExit("No games retrieved.")
    return pd.concat(frames, ignore_index=True)


def resolve_file_input(parser: argparse.ArgumentParser, args) -> Optional[str]:
    if args.live:
        return None

    if args.file:
        csv_path = Path(args.file).expanduser()
        if not csv_path.exists():
            parser.error(f"CSV file not found: {csv_path}")
        return str(csv_path)

    csv_candidates = sorted(Path.cwd().glob("*.csv"))
    if len(csv_candidates) == 1:
        csv_path = csv_candidates[0]
        print(f"  Auto-detected CSV input: {csv_path}")
        return str(csv_path)

    if not csv_candidates:
        parser.error("provide --file PATH or --live; no CSV files were found in the current directory")

    csv_names = ", ".join(path.name for path in csv_candidates)
    parser.error(f"multiple CSV files found in the current directory; use --file PATH to choose one: {csv_names}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HoopIQ — Apriori play-sequence mining for NBA data"
    )
    # Data source
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file",   type=str, help="Path to CSV file (nba_api PlayByPlayV2 schema)")
    src.add_argument("--live",   action="store_true", help="Fetch live from NBA Stats API")

    # Live options
    parser.add_argument("--team",   type=str, default="BOS", help="Team abbreviation (live mode)")
    parser.add_argument("--season", type=str, default="2023-24", help="Season (live mode)")
    parser.add_argument("--games",  type=int, default=20, help="Number of games to fetch (live)")

    # Algorithm params
    parser.add_argument("--min-support",    type=float, default=0.05,
                        help="Minimum support threshold (default 0.05 = 5%%)")
    parser.add_argument("--min-confidence", type=float, default=0.30,
                        help="Minimum confidence threshold (default 0.30)")
    parser.add_argument("--min-lift",       type=float, default=1.0,
                        help="Minimum lift (default 1.0 — no filter)")
    parser.add_argument("--max-seq-len",    type=int,   default=5,
                        help="Max events per sequence (default 5)")
    parser.add_argument("--min-seq-len",    type=int,   default=2,
                        help="Min events per sequence (default 2)")
    parser.add_argument("--top-n",          type=int,   default=20,
                        help="Number of top rules to display (default 20)")

    # Output
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for exported CSV/JSON (default: current dir)")
    parser.add_argument("--no-export",  action="store_true",
                        help="Skip file export (console output only)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip support threshold sweep")

    args = parser.parse_args()
    resolved_file = resolve_file_input(parser, args)

    T0 = time.perf_counter()

    banner("HoopIQ — Play Sequence Pattern Mining  |  Apriori Engine")
    print(f"  min_support={args.min_support:.0%}  "
          f"min_confidence={args.min_confidence:.0%}  "
          f"min_lift={args.min_lift}  "
          f"max_seq_len={args.max_seq_len}")

    # ── 1. Load data ────────────────────────────────────────────────────────────
    print("\n  Loading data ...", end=" ", flush=True)
    t = time.perf_counter()

    if args.live:
        pbp_df = fetch_live(args.team, args.season, args.games)
    else:
        pbp_df = pd.read_csv(resolved_file)
        # Normalise column names (some exports use lowercase)
        pbp_df.columns = [c.upper() for c in pbp_df.columns]

    print(f"{len(pbp_df):,} events across {pbp_df['GAME_ID'].nunique()} game(s)  "
          f"[{time.perf_counter()-t:.2f}s]")

    # ── 2. Extract sequences ────────────────────────────────────────────────────
    print("  Extracting possession sequences ...", end=" ", flush=True)
    t = time.perf_counter()
    all_seqs, scoring_seqs = extract_sequences(
        pbp_df,
        min_len=args.min_seq_len,
        max_len=args.max_seq_len,
    )
    print(f"{len(all_seqs):,} total  |  {len(scoring_seqs):,} scoring  "
          f"[{time.perf_counter()-t:.3f}s]")

    if not all_seqs:
        raise SystemExit("No sequences extracted. Check column names in your CSV.")

    # ── 3. Print summaries ──────────────────────────────────────────────────────
    print_dataset_summary(pbp_df, all_seqs, scoring_seqs)
    print_event_distribution(all_seqs)
    print_scoring_breakdown(scoring_seqs)
    print_top_sequences(all_seqs, scoring_seqs)

    # ── 4. Benchmark sweep ──────────────────────────────────────────────────────
    if not args.no_benchmark:
        print("\n  Running threshold sweep ...", flush=True)
        sweep_df = sweep_thresholds(all_seqs, scoring_seqs)
        print_benchmark(sweep_df)

    # ── 5. Main Apriori run ─────────────────────────────────────────────────────
    print(f"\n  Running Apriori (support={args.min_support:.0%}) ...", end=" ", flush=True)
    rules_df, metrics = run_apriori_pipeline(
        all_seqs, scoring_seqs,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_seq_len,
    )
    print(f"{metrics.get('n_freq_itemsets',0)} itemsets → "
          f"{metrics.get('n_scoring_rules',0)} scoring rules  "
          f"[{metrics.get('total_time',0):.3f}s]")

    print_top_rules(rules_df, n=args.top_n)

    # ── 6. Export ────────────────────────────────────────────────────────────────
    if not args.no_export:
        export_results(rules_df, all_seqs, scoring_seqs, output_dir=args.output_dir)

    total = time.perf_counter() - T0
    banner(f"DONE  —  total runtime {total:.2f}s")
    print(f"""
  Next steps:
    • Tune --min-support and --min-confidence to surface more/fewer rules
    • Use hoopiq_patterns.json as the data contract for your web app API
    • Add --team / --season flags with --live to pull fresh NBA data
    • Feed hoopiq_results.csv into a visualisation layer (D3 / Recharts)
""")


if __name__ == "__main__":
    main()
