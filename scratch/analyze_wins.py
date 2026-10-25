import sqlite3, json

conn = sqlite3.connect('data/bot_memory.sqlite3')
cur = conn.cursor()
wins = cur.execute('SELECT id, symbol, direction, pnl, score, confidence, indicators, components FROM trades WHERE pnl > 0 ORDER BY id DESC LIMIT 20').fetchall()
print(f'Total recent wins inspected: {len(wins)}')
for w in wins:
    comps = json.loads(w[7] or '{}')
    inds = json.loads(w[6] or '{}')
    rsi = inds.get('rsi', 0)
    adx = inds.get('adx', 0)
    print(f'Win  #{w[0]} | {w[1]:<10} {w[2]:<5} | PnL: +${w[3]:.2f} | score={w[4]} conf={w[5]} rsi={rsi:.1f} adx={adx:.1f} | comps={comps}')
