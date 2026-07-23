.PHONY: help install train train-course train-full finetune eval \
        benchmark test test-all clean clean-all plot serve lint \
        gradio gradio-share hf-convert hf-verify hf-push

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install in dev mode with all extras
	pip install -e ".[all]"

train:  ## Run full training pipeline (alias for train-full)
	$(MAKE) train-full

train-course:  ## Fast course training (~100 utts, pipeline verification)
	bash scripts/03_train_course_fast.sh

train-full:  ## Full AISHELL-1 training (141k utts, 360 epochs, GPU)
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

test:  ## Run pytest on CER + data-loading tests
	python -m pytest tests/test_cer.py tests/test_data_loading.py -v --tb=short -x

test-all:  ## Run all tests including slow/fbank integration tests
	python -m pytest tests/test_cer.py tests/test_data_loading.py tests/test_fbank.py -v --tb=short -x

plot:  ## Generate publication-quality benchmark charts
	python benchmark/plot_results.py --output figures
	python benchmark/gen_arch_diagram.py

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

clean:  ## Remove regenerable artifacts (figures, cache)
	rm -rf figures/ __pycache__ .pytest_cache
	@echo "  results/ is preserved (contains validated benchmark data)"
	@echo "  Use 'make clean-all' to also remove results/ if needed"

clean-all:  ## Remove all generated data including validated results
	rm -rf results/*.md results/*.csv figures/ __pycache__ .pytest_cache
	@echo "  results/ cleared. Re-run benchmark scripts to regenerate."
