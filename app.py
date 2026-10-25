"""
app.py - Web Dashboard & Background Bot Manager for Render / Cloud Hosting.

Runs an interactive status dashboard on the cloud PORT while keeping the
Deriv trading bot running 24/7 in the background with dynamic Mode Switching.
"""
import os
import sys
import subprocess
import threading
import time
import sqlite3
import urllib.parse
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Global state
bot_process = None
active_mode = os.getenv("TRADE_MODE", "hybrid").strip().lower()

def start_bot_process(mode=None):
    global bot_process, active_mode
    if mode:
        active_mode = mode.strip().lower()
        os.environ["TRADE_MODE"] = active_mode

    # Cleanly terminate existing process if running
    if bot_process is not None and bot_process.poll() is None:
        try:
            bot_process.terminate()
            bot_process.wait(timeout=5)
        except Exception:
            try:
                bot_process.kill()
            except Exception:
                pass

    # Clear stale instance locks unconditionally
    for lock_path in ("data/bot.lock", "data/bot_memory.sqlite3-wal", "STOP_AFTER_TRADE"):
        if os.path.exists(lock_path):
            try:
                os.unlink(lock_path)
            except Exception:
                pass

    cmd = [sys.executable, "main.py", "--mode", active_mode, "--force-lock"]
    os.makedirs("logs", exist_ok=True)
    log_file = open("logs/bot.log", "a")
    bot_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Trading bot started in [{active_mode.upper()}] mode with PID {bot_process.pid}")
    return f"Bot running in [{active_mode.upper()}] mode (PID {bot_process.pid})"

def bot_supervisor():
    """Continuous watchdog to ensure bot stays alive 24/7."""
    time.sleep(10)
    while True:
        try:
            if bot_process is None or bot_process.poll() is not None:
                exit_code = bot_process.poll() if bot_process else "None"
                print(f"[WATCHDOG] Bot process down (code: {exit_code}). Auto-restarting in {active_mode} mode...")
                start_bot_process(active_mode)
        except Exception as e:
            print(f"[WATCHDOG] Exception: {e}")
        time.sleep(10)

def get_recent_logs(lines=40):
    log_path = "logs/bot.log"
    if not os.path.exists(log_path):
        return "Log file not created yet. Waiting for bot initialization..."
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading logs: {e}"

def get_trade_summary():
    db_path = "data/bot_memory.sqlite3"
    if not os.path.exists(db_path):
        return "No trades recorded yet."
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT id, symbol, direction, contract_type, status, pnl, score, regime, entry_time FROM trades ORDER BY id DESC LIMIT 15", conn)
        conn.close()
        if df.empty:
            return "No trades in database yet."
        df['time'] = pd.to_datetime(df['entry_time'], unit='s').dt.strftime('%H:%M:%S')
        df['pnl'] = df['pnl'].apply(lambda x: f"${x:+.2f}" if pd.notnull(x) else "open")
        display_cols = ['id', 'time', 'symbol', 'direction', 'contract_type', 'status', 'pnl', 'score']
        return df[display_cols].to_string(index=False)
    except Exception as e:
        return f"Error querying database: {e}"

# Start bot initially in background thread with supervisor watchdog
threading.Thread(target=start_bot_process, daemon=True).start()
threading.Thread(target=bot_supervisor, daemon=True).start()

# Web Server (Listens on Render $PORT or standard 7860/10000)
port = int(os.environ.get("PORT", 7860))

try:
    import gradio as gr

    with gr.Blocks(title="Deriv Autonomous Trading Bot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Deriv Autonomous Trading Bot")
        gr.Markdown("24/7 Cloud Execution with Live Strategy Switching and Confluence Engine.")

        with gr.Row():
            mode_dropdown = gr.Dropdown(
                choices=["rise_fall", "multiplier", "hybrid"],
                value=active_mode,
                label="Active Trading Mode"
            )
            switch_btn = gr.Button("⚡ Switch Mode & Restart Bot", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh Live Feed")

        status_text = gr.Textbox(
            value=f"🟢 Engine Active in [{active_mode.upper()}] Mode",
            label="System Status",
            interactive=False
        )

        with gr.Tab("Live Console Logs"):
            log_output = gr.Code(value=get_recent_logs, label="Recent bot.log Output", language="shell", every=10)

        with gr.Tab("Recent Trades"):
            trades_output = gr.Code(value=get_trade_summary, label="Database Closed & Open Contracts", every=15)

        switch_btn.click(
            fn=lambda m: (start_bot_process(m), get_recent_logs(), get_trade_summary()),
            inputs=[mode_dropdown],
            outputs=[status_text, log_output, trades_output]
        )
        refresh_btn.click(
            fn=lambda: (get_recent_logs(), get_trade_summary()),
            outputs=[log_output, trades_output]
        )

    if __name__ == "__main__":
        demo.queue().launch(server_name="0.0.0.0", server_port=port)

except ImportError:
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class StatusHandler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)

            # Handle 1-click mode switch from URL (e.g. /?mode=multiplier)
            msg = ""
            if "mode" in query:
                req_mode = query["mode"][0].lower()
                if req_mode in ("rise_fall", "multiplier", "hybrid"):
                    msg = f"<div style='background:#1b5e20;padding:10px;border-radius:4px;margin-bottom:15px;'>✅ {start_bot_process(req_mode)}</div>"

            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            logs = get_recent_logs(40)
            trades = get_trade_summary()
            is_running = bot_process is not None and bot_process.poll() is None
            if is_running:
                status_badge = f"<span style='color: #4ade80; font-weight: bold;'>🟢 ACTIVE & SCANNING (PID: {bot_process.pid})</span>"
            else:
                code = bot_process.poll() if bot_process else "None"
                status_badge = f"<span style='color: #f87171; font-weight: bold;'>🔴 RESTARTING (Exit code: {code})</span>"

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Deriv Trading Bot (Live)</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <meta http-equiv="refresh" content="25">
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #0f172a; color: #f8fafc; padding: 20px; }}
                    .btn {{ display: inline-block; padding: 10px 18px; margin: 5px; font-weight: bold; border-radius: 6px; text-decoration: none; color: white; }}
                    .btn-rf {{ background: #2563eb; }}
                    .btn-mult {{ background: #7c3aed; }}
                    .btn-hyb {{ background: #059669; }}
                    .btn:hover {{ opacity: 0.85; }}
                    pre {{ background: #1e293b; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.4; }}
                </style>
            </head>
            <body>
                <h2>🚀 Deriv Autonomous Trading Bot</h2>
                <p>Status: {status_badge} | Mode: <strong style="color: #38bdf8; text-transform: uppercase;">{active_mode}</strong></p>
                {msg}
                <div style="margin: 20px 0;">
                    <strong>Switch Mode (1-Click):</strong><br>
                    <a class="btn btn-rf" href="/?mode=rise_fall">⚡ Switch to RISE_FALL</a>
                    <a class="btn btn-mult" href="/?mode=multiplier">⚡ Switch to MULTIPLIER</a>
                    <a class="btn btn-hyb" href="/?mode=hybrid">⚡ Switch to HYBRID</a>
                </div>
                <h3>📊 Recent Trades:</h3>
                <pre>{trades}</pre>
                <h3>📜 Live Console Logs (Auto-refresh 25s):</h3>
                <pre>{logs}</pre>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

    server = HTTPServer(("0.0.0.0", port), StatusHandler)
    server.serve_forever()
