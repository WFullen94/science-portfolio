.PHONY: setup ground pack test clean

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip wheel && \
		python -m pip install -r requirements.txt && python -m pip install -e .

# Phase 1 — the grounded generator (works)
ground: ; . .venv/bin/activate && python -m detsynth.stix           # STIX -> grounding
pack:   ; . .venv/bin/activate && python -m detsynth.generate $(TID) # e.g. make pack TID=T1053.005

# Phase 2 — DPO faithfulness (code ready; blocked on a pinned TRL/transformers stack)
align-data: ; . .venv/bin/activate && python -m detsynth.align_data
sft:        ; . .venv/bin/activate && python -m detsynth.sft
dpo:        ; . .venv/bin/activate && python -m detsynth.dpo
faithful:   ; . .venv/bin/activate && python -m detsynth.evaluate

test:   ; . .venv/bin/activate && pytest -q                          # grounding + format (no model)

clean:
	rm -rf data/packs data/ckpt_* data/align_prefs data/reports data/*.jsonl mlflow.db .pytest_cache
