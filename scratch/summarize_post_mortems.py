import sqlite3
import json
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('data/bot_memory.sqlite3')
df_pm = pd.read_sql_query('SELECT * FROM loss_post_mortems', conn)
df_trades = pd.read_sql_query('SELECT * FROM trades', conn)
conn.close()

print(f"Total loss post mortems: {len(df_pm)}")
categories = {}
root_causes = []
symbols = {}

for idx, row in df_pm.iterrows():
    cat = row.get('category')
    sym = row.get('symbol')
    rc = row.get('root_cause')
    patch = row.get('strategy_patch')
    dt = datetime.fromtimestamp(row.get('created_at', 0)).strftime('%Y-%m-%d %H:%M') if row.get('created_at') else 'N/A'
    
    categories[cat] = categories.get(cat, 0) + 1
    symbols[sym] = symbols.get(sym, 0) + 1
    root_causes.append((dt, sym, row.get('direction'), cat, rc, patch))

print("\n--- Failure Categories ---")
for c, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"  {c}: {cnt}")

print("\n--- Losses by Symbol ---")
for s, cnt in sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {s}: {cnt}")

print("\n--- Detailed Post-Mortem Analysis (Recent 20) ---")
for dt, sym, d, cat, rc, patch in root_causes[-20:]:
    print(f"[{dt}] {sym} ({d}) -> {cat}")
    print(f"  Root Cause: {rc}")
    print(f"  Patch: {patch}")
    print()
