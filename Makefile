# Deriv Bot - Make targets
VENV    = .venv
PYTHON  = $(VENV)/bin/python
PROJ    = /Users/aditya/Desktop/TYD/Claude_Ai_Trade

.PHONY: install preflight dry-run run kill-switch

install:
	@echo "Creating venv and installing deps (numba skipped for Python 3.14+)..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install "websockets>=12.0" "pandas>=2.2" "numpy>=1.26" "python-dotenv>=1.0" tqdm
	$(VENV)/bin/pip install "pandas-ta>=0.4.67b0" --no-deps
	$(PYTHON) patch_pandas_ta.py $(VENV)/lib
	@echo "Done. Copy .env.example to .env and add your token."

preflight:
	$(PYTHON) main.py --preflight

dry-run:
	$(PYTHON) main.py --dry-run

run:
	@echo "WARNING: This sends real orders on the DEMO account."
	@read -p "Press Enter to continue or Ctrl-C to abort..."
	$(PYTHON) main.py

kill-switch:
	touch KILL_SWITCH
	@echo "Kill switch SET. Bot will veto new signals."

unkill:
	rm -f KILL_SWITCH
	@echo "Kill switch CLEARED."
