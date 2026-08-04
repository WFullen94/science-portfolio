.PHONY: setup data xgb ft compare test clean

# One-shot venv + deps.
setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip wheel && \
		python -m pip install -r requirements.txt && python -m pip install -e .

# --- Stage 1: tabular (FT-Transformer vs XGBoost) ----------------------------
data:      ; . .venv/bin/activate && python -m structdl.data
xgb:       ; . .venv/bin/activate && python -m structdl.xgb_baseline
ft:        ; . .venv/bin/activate && python -m structdl.ft_transformer
compare:   ; . .venv/bin/activate && python -m structdl.compare      # tabular head-to-head

# --- Stage 2: sequence (time-series Transformer vs GRU) ----------------------
seqdata:   ; . .venv/bin/activate && python -m structdl.sequence_data
gru:       ; . .venv/bin/activate && python -m structdl.gru_baseline
ts:        ; . .venv/bin/activate && python -m structdl.ts_transformer
ts-compare:; . .venv/bin/activate && python -m structdl.ts_compare    # sequence head-to-head
ts-sweep:  ; . .venv/bin/activate && python -m structdl.ts_sweep      # length sweep (headline)

# --- Stage 3: distributed training (DDP runs here on CPU/gloo; FSDP -> CUDA) --
ddp:       ; . .venv/bin/activate && python -m structdl.train_distributed --strategy ddp
fsdp:      ; . .venv/bin/activate && python -m structdl.train_distributed --strategy fsdp

test:      ; . .venv/bin/activate && pytest -q

clean:
	rm -rf data/*.parquet data/reports mlflow.db mlartifacts .pytest_cache
