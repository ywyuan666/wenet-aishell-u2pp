# Architecture Comparison Results


## decode_mode

| Variant | CER(%) | RTF | Latency(ms) | Params |
|---------|--------|-----|------------|--------|
| ctc_greedy | 100.0 | 0.161 | inf | {'modes': ['ctc_greedy_search']} |
| ctc_prefix_beam | FAIL | N/A | N/A | {'modes': ['ctc_prefix_beam_search'], 'beam_size': 5} |
| attention_rescore | FAIL | N/A | N/A | {'modes': ['attention_rescoring'], 'beam_size': 5} |

## chunk_size

| Variant | CER(%) | RTF | Latency(ms) | Params |
|---------|--------|-----|------------|--------|
| non_streaming | 100.0 | 0.158 | inf | {'chunk_size': -1} |
| chunk_16 | 100.0 | 0.158 | 640 | {'chunk_size': 16} |
| chunk_8 | 100.0 | 0.156 | 320 | {'chunk_size': 8} |
| chunk_4 | 100.0 | 0.157 | 160 | {'chunk_size': 4} |

## ctc_weight

| Variant | CER(%) | RTF | Latency(ms) | Params |
|---------|--------|-----|------------|--------|
| ctc_0.1 | 100.0 | 0.156 | inf | {'ctc_weight': 0.1} |
| ctc_0.3 | 100.0 | 0.157 | inf | {'ctc_weight': 0.3} |
| ctc_0.5 | 100.0 | 0.155 | inf | {'ctc_weight': 0.5} |
| ctc_0.7 | 100.0 | 0.154 | inf | {'ctc_weight': 0.7} |