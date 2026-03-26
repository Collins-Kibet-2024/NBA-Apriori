# 🏀 HoopIQ — NBA Play Sequence Pattern Mining

HoopIQ applies the **Apriori association rule mining algorithm** to NBA play-by-play data to discover offensive sequences that lead to scoring events. By treating each possession as a transaction, the system mines patterns like *"offensive rebound → kick-out → 3-pointer"* and ranks them by support, confidence, and lift — surfacing the plays a team runs most reliably and most efficiently.

The project consists of a Python data pipeline (`hoopiq_apriori.py`) that ingests live NBA data via the `nba_api` package or any `PlayByPlayV2`-schema CSV, and a standalone web dashboard (`hoopiq_app.html`) where coaches can explore top patterns, filter by play style and game situation, and get ranked play suggestions with coaching notes — no server required.
