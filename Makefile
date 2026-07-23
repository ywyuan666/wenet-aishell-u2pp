.PHONY: help train train-course train-full finetune eval benchmark lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

lint:  ## Check shell and Python syntax
	bash -n env_autodl.sh
	for f in scripts/*.sh; do bash -n "$$f"; done
	python -m py_compile eval_cer.py run_eval.py
	for f in benchmark/*.py; do python -m py_compile "$$f"; done
	for f in tools/*.py; do python -m py_compile "$$f"; done

clean:  ## Remove generated results
	rm -rf results/*.md results/*.csv
