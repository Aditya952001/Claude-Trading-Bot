"""
monitor_render_bot.py - Autonomous 24/7 Monitor & Loss Forensic Watcher.

Continuously polls https://claude-trading-bot-kc7x.onrender.com for:
1. Live Engine & Process Health (is it active & scanning 38+ symbols?)
2. Real-time Trade Executions (Wins & Losses)
3. Automated Forensic Post-Mortem on any loss
"""
import time
import json
import urllib.request
import sys
from datetime import datetime

URL = "https://claude-trading-bot-kc7x.onrender.com"

def fetch_json(endpoint):
    try:
        req = urllib.request.Request(f"{URL}/?api={endpoint}", headers={"User-Agent": "AutonomousMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return None

def fetch_logs():
    try:
        req = urllib.request.Request(f"{URL}/?api=logs", headers={"User-Agent": "AutonomousMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode(errors="ignore")
    except Exception as e:
        return ""

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Autonomous Monitor started against {URL}", flush=True)
    seen_trade_ids = set()
    initial_check = True

    while True:
        try:
            trades = fetch_json("trades")
            if trades and isinstance(trades, list) and len(trades) > 0 and "error" not in trades[0]:
                for t in trades:
                    tid = t.get("id")
                    if tid not in seen_trade_ids:
                        seen_trade_ids.add(tid)
                        pnl = float(t.get("pnl") or 0)
                        sym = t.get("symbol")
                        direction = t.get("direction")
                        ctype = t.get("contract_type")
                        score = t.get("score")
                        if not initial_check:
                            if pnl > 0:
                                print(f"🟢 [NEW WIN] Trade #{tid} | {sym} {direction} {ctype} | PnL: +${pnl:.2f} | Score: {score}", flush=True)
                            else:
                                print(f"🚨 [NEW LOSS] Trade #{tid} | {sym} {direction} {ctype} | PnL: -${abs(pnl):.2f} | Score: {score}", flush=True)
                                pms = fetch_json("post_mortems")
                                if pms and isinstance(pms, list):
                                    for pm in pms:
                                        if pm.get("trade_id") == tid:
                                            print(f"   [POST-MORTEM]: {json.dumps(pm, indent=2)}", flush=True)
                        else:
                            print(f"   Recorded Trade #{tid}: {sym} {direction} {ctype} | PnL: ${pnl:+.2f}", flush=True)
                initial_check = False

            logs = fetch_logs()
            lines = logs.splitlines()
            for line in reversed(lines[-15:]):
                if any(k in line for k in ("LIVE ACTIVE POSITIONS", "BAR CLOSE", "AUDIT UPDATE", "Nominal")):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}", flush=True)
                    break
        except Exception as e:
            print(f"\n[Monitor Error]: {e}", flush=True)
        time.sleep(20)

if __name__ == "__main__":
    main()
