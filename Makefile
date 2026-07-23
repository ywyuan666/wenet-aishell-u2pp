.PHONY: help train train-course train-full finetune eval benchmark \
        test plot serve install lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install in dev mode with all extras
	pip install -e ".[all]"

train-course:  ## Fast course training (3000 utts, quick iteration)
	bash scripts/03_train_course_fast.sh

train-full:  ## Full AISHELL-1 training (120k utts)
	bash scripts/03_train_full.sh

finetune:  ## Fine-tune from a checkpoint (set CHECKPOINT env var)
	bash scripts/03_finetune_from_ckpt.sh

eval:  ## Decode + CER evaluation
	bash scripts/04_decode_eval.sh

benchmark:  ## Run all benchmark scripts
	python benchmark/compare_architectures.py
	python benchmark/streaming_tradeoff.py
	python benchmark/error_analysis.py
	python benchmark/quantize_and_demo.py

test:  ## Run pytest on CER / data-loading / fbank tests
	python -m pytest tests/ -v --tb=short -x

test-all:  ## Run all tests including slow integration tests
	python -m pytest tests/ -v --tb=short --runslow

plot:  ## Generate benchmark visualization charts
	python benchmark/plot_results.py --output figures

serve:  ## Start streaming ASR server (chunk=16, port=8765)
	python asr_server.py --chunk 16 --port 8765

gradio:  ## Start Gradio Web UI (browser-based ASR demo)
	python app_gradio.py --port 7860

gradio-share:  ## Start Gradio with public share link
	python app_gradio.py --port 7860 --share

hf-convert:  ## Convert checkpoint to HuggingFace format
	python scripts/convert_to_hf.py --output hf_model

hf-verify:  ## Verify HuggingFace model
	python scripts/convert_to_hf.py --verify hf_model

hf-push:  ## Push HuggingFace model to Hub (set HF_TOKEN)
	python scripts/convert_to_hf.py --output hf_model --push-to-hub $(HF_REPO)

lint:  ## Check shell and Python syntax
	bash -n env_autodl.sh
	for f in scripts/*.sh; do bash -n "$$f"; done
	-flake8 eval_cer.py run_eval.py tools/ benchmark/ --config=.flake8
	python -m pytest tests/ --collect-only -q 2>/dev/null || true

clean:  ## Remove generated results and artifacts
	rm -rf results/*.md results/*.csv figures/ __pycache__ .pytest_cache
