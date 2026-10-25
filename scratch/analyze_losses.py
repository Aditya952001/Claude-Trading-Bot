import sqlite3
import json
import pandas as pd

conn = sqlite3.connect('data/bot_memory.sqlite3')
df_trades = pd.read_sql_query('SELECT * FROM trades', conn)
df_pm = pd.read_sql_query('SELECT * FROM loss_post_mortems', conn)
conn.close()

print(f"Total trades in DB: {len(df_trades)}")
if not df_trades.empty:
    print("Statuses:", df_trades['status'].value_counts().to_dict())
    settled = df_trades[df_trades['status'].isin(['won', 'lost', 'closed'])]
    wins = len(df_trades[df_trades['status'] == 'won'])
    losses = len(df_trades[df_trades['status'] == 'lost'])
    if wins + losses > 0:
        print(f"Wins: {wins}, Losses: {losses}, Win rate: {wins / (wins + losses):.1%}")
    if 'pnl' in df_trades.columns:
        print(f"Total PnL: ${df_trades['pnl'].dropna().sum():.2f}")

print("\n--- Recent Trades (last 25) ---")
cols = ['id', 'symbol', 'direction', 'status', 'pnl', 'entry_time', 'exit_time']
available_cols = [c for c in cols if c in df_trades.columns]
print(df_trades[available_cols].tail(25).to_string())

print(f"\n--- Loss Post-Mortems Columns: {list(df_pm.columns)} ---")
if not df_pm.empty:
    print(df_pm.tail(10).to_string())

