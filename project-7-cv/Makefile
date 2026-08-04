.PHONY: setup verify sample train eval export serve test clean

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip wheel && \
		python -m pip install -r requirements.txt && python -m pip install -e .

verify:  ; . .venv/bin/activate && python -m overwatch.data          # check dataset + build sample
train:   ; . .venv/bin/activate && python -m overwatch.train          # fine-tune YOLOv8n on VisDrone
eval:    ; . .venv/bin/activate && python -m overwatch.evaluate       # untiled vs SAHI head-to-head
export:  ; . .venv/bin/activate && python -m overwatch.export_onnx    # -> data/overwatch.onnx
serve:   ; . .venv/bin/activate && uvicorn overwatch.serve:app --reload
test:    ; . .venv/bin/activate && pytest -q                          # mAP metric tests (no data/model)

docker-build: ; docker build -f docker/Dockerfile -t overwatch .

clean:
	rm -rf data/runs data/reports data/sample data/*.onnx mlflow.db .pytest_cache
