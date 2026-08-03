.PHONY: setup data xgb ft compare test clean

# One-shot venv + deps.
setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip wheel && \
		python -m pip install -r requirements.txt && python -m pip install -e .

data:      ; . .venv/bin/activate && python -m structdl.data
xgb:       ; . .venv/bin/activate && python -m structdl.xgb_baseline
ft:        ; . .venv/bin/activate && python -m structdl.ft_transformer
compare:   ; . .venv/bin/activate && python -m structdl.compare   # the head-to-head
test:      ; . .venv/bin/activate && pytest -q

clean:
	rm -rf data/*.parquet data/reports mlflow.db mlartifacts .pytest_cache
