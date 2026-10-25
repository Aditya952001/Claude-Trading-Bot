import sqlite3

conn = sqlite3.connect('data/bot_memory.sqlite3')
cur = conn.cursor()
trades = cur.execute('SELECT id, symbol, direction, entry_price, exit_price, pnl, datetime(entry_time, "unixepoch", "localtime"), datetime(exit_time, "unixepoch", "localtime") FROM trades WHERE entry_time >= 1792313200').fetchall()

print('=== TRADES SINCE AI FIX (14:27) ===')
wins = [t for t in trades if t[5] is not None and t[5] > 0]
losses = [t for t in trades if t[5] is not None and t[5] < 0]
open_t = [t for t in trades if t[5] is None]
gross_profit = sum(t[5] for t in wins)
gross_loss = abs(sum(t[5] for t in losses))
net_pnl = gross_profit - gross_loss
wr = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

for t in trades:
    status = f'+${t[5]:.2f}' if t[5] is not None and t[5] > 0 else (f'-${abs(t[5]):.2f}' if t[5] is not None else 'OPEN')
    print(f'#{t[0]} | {t[1]:<10} | {t[2]:<5} | Entry: {t[3]} | Exit: {t[4]} | PnL: {status} | Time: {t[6]}')

print('\n=== SUMMARY METRICS ===')
print(f'Total Closed Trades: {len(wins) + len(losses)}')
print(f'Wins: {len(wins)} | Losses: {len(losses)} | Open: {len(open_t)}')
print(f'Win Rate: {wr:.1f}%')
print(f'Gross Profit: +${gross_profit:.2f}')
print(f'Gross Loss:   -${gross_loss:.2f}')
print(f'Net PnL:      +${net_pnl:.2f}')
print(f'Profit Factor: {pf:.2f}')
