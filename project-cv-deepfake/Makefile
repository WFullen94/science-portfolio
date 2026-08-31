.PHONY: setup data train eval export serve test docker-build clean

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip wheel && \
		python -m pip install -r requirements.txt && python -m pip install -e .

data:   ; . .venv/bin/activate && python -m dfdetect.data            # download + verify + CI sample
train:  ; . .venv/bin/activate && python -m dfdetect.train           # fine-tune EfficientNet-B0
eval:   ; . .venv/bin/activate && python -m dfdetect.evaluate        # score on held-out test.zip
export: ; . .venv/bin/activate && python -m dfdetect.export_onnx     # -> data/dfdetect.onnx
serve:  ; . .venv/bin/activate && uvicorn dfdetect.serve:app --reload
test:   ; . .venv/bin/activate && pytest -q                          # split/leakage tests (no model)

docker-build: ; docker build -f docker/Dockerfile -t dfdetect .

clean:
	rm -rf data/raw data/ckpt data/reports data/sample data/*.onnx mlflow.db .pytest_cache
