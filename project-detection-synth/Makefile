.PHONY: setup ground pack test clean

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip wheel && \
		python -m pip install -r requirements.txt && python -m pip install -e .

# Phase 1 — the grounded generator (works)
ground: ; . .venv/bin/activate && python -m detsynth.stix           # STIX -> grounding
pack:   ; . .venv/bin/activate && python -m detsynth.generate $(TID) # e.g. make pack TID=T1053.005

# Phase 2 — DPO faithfulness (needs requirements-align.txt — see its header)
align-data: ; . .venv/bin/activate && python -m detsynth.align_data
sft:        ; . .venv/bin/activate && python -m detsynth.sft
dpo:        ; . .venv/bin/activate && python -m detsynth.dpo
faithful:   ; . .venv/bin/activate && python -m detsynth.evaluate

# Capstone — incident -> technique mapping -> incident-conditioned pack
map:            ; . .venv/bin/activate && python -m detsynth.mapping --incident "$(INCIDENT)"
incident-eval:  ; . .venv/bin/activate && python -m detsynth.incident_eval
incident-pack:  ; . .venv/bin/activate && python -m detsynth.incident_pack --technique-id $(TID) --incident "$(INCIDENT)"
serve-incident: ; . .venv/bin/activate && uvicorn detsynth.serve:app --reload

test:   ; . .venv/bin/activate && pytest -q                          # grounding/format/mapping (no model)

clean:
	rm -rf data/packs data/ckpt_* data/align_prefs data/reports data/*.jsonl mlflow.db .pytest_cache
