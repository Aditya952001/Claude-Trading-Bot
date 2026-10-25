import sqlite3
import pandas as pd

conn = sqlite3.connect("data/bot_memory.sqlite3")
df = pd.read_sql_query("SELECT * FROM trades WHERE status='closed' AND contract_type='multiplier' AND pnl IS NOT NULL", conn)
wins = df[df["pnl"] > 0]
losses = df[df["pnl"] < 0]

avg_win = wins["pnl"].mean()
avg_loss = losses["pnl"].mean()
print(f"Multiplier Wins: {len(wins)}, Avg Win: ${avg_win:.3f}")
print(f"Multiplier Losses: {len(losses)}, Avg Loss: ${avg_loss:.3f}")
print("\nTop 5 Loss Exit Reasons:")
print(losses["exit_reason"].value_counts().head(5))
print("\nTop 5 Win Exit Reasons:")
print(wins["exit_reason"].value_counts().head(5))
