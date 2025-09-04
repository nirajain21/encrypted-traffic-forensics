.PHONY: env check train rf pipeline ustc clean

env:
	python3 -m venv cfm-env && . cfm-env/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

check:
	bash scripts/check_deps.sh

train:
	python3 src/train_models.py

rf:
	python3 src/train_rf.py

pipeline:
	python3 src/pipeline_rf.py

ustc:
	python3 src/ustc_eval.py

clean:
	rm -rf models/* figures/* flows/* 2>/dev/null || true
	mkdir -p models figures flows && touch flows/.gitkeep
