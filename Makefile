.PHONY: help install test lint data demo ofi mm bench all

help:
	@echo "make install   install dependencies"
	@echo "make test      run the test suite"
	@echo "make lint      run ruff"
	@echo "make data      generate the synthetic capture"
	@echo "make demo      walk one order through the whole stack"
	@echo "make ofi       replicate the order flow imbalance result"
	@echo "make mm        compare inventory-aware vs symmetric quoting"
	@echo "make bench     measure the tick-to-order latency path"
	@echo "make all       data + demo + ofi + mm + bench"

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

data:
	python scripts/gen_synthetic.py --events 200000

demo:
	python scripts/demo_oms.py

ofi: data
	python scripts/analyse_ofi.py --capture data/synthetic_okx.jsonl.gz --plot docs/ofi_scatter.png

mm:
	python scripts/run_mm.py --runs 5000 --plot docs/mm_pnl.png

bench:
	python scripts/bench_latency.py --messages 100000

all: data demo ofi mm bench
