#!/usr/bin/env python3
"""P0: Architecture Comparison - Conformer vs Transformer vs Ablation Study.
Generates config files and runs decode comparison with different architectures,
decode modes, and chunk sizes. Outputs a comparison table in CSV format."""

import torch, yaml, json, os, sys, soundfile, time, csv
from pathlib import Path
sys.path.insert(0, r'D:\wenet\wenet')
os.chdir(r'D:\wenet\wenet\examples\aishell\s0')

if not hasattr(torch.nn.Module, '__annotations__'):
    torch.nn.Module.__annotations__ = {}

from wenet.utils.init_model import init_model
from wenet.text.char_tokenizer import CharTokenizer
from torchaudio.compliance import kaldi

# --- Configs to compare ---
COMPARISONS = {
    'decode_mode': {
        'ctc_greedy':        {'modes': ['ctc_greedy_search']},
        'ctc_prefix_beam':   {'modes': ['ctc_prefix_beam_search'], 'beam_size': 5},
        'attention_rescore': {'modes': ['attention_rescoring'], 'beam_size': 5},
    },
    'chunk_size': {
        'non_streaming': {'chunk_size': -1},
        'chunk_16':      {'chunk_size': 16},
        'chunk_8':       {'chunk_size': 8},
        'chunk_4':       {'chunk_size': 4},
    },
    'ctc_weight': {
        'ctc_0.1': {'ctc_weight': 0.1},
        'ctc_0.3': {'ctc_weight': 0.3},
        'ctc_0.5': {'ctc_weight': 0.5},
        'ctc_0.7': {'ctc_weight': 0.7},
    },
}

# --- Load model ---
class Args: pass
args = Args(); args.jit = True
args.checkpoint = 'exp/u2pp_conformer_course/epoch_4.pt'
args.config = 'exp/u2pp_conformer_course/train.yaml'

with open(args.config) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
model, configs = init_model(args, configs)
model.eval()

tokenizer = CharTokenizer('data/dict/lang_char.txt', non_lang_syms=None)

# --- Load test data ---
test_items = []
with open('data/test/data.list', encoding='utf-8') as f:
    for line in f:
        test_items.append(json.loads(line.strip()))
sample_n = min(10, len(test_items))  # small sample for comparison
print(f'Comparing on {sample_n} test utterances\n')

# --- Pre-extract fbank features ---
def extract_fbank(wav_path):
    wav, sr = soundfile.read(wav_path, dtype='float32')
    wav = torch.from_numpy(wav).unsqueeze(0)
    if sr != 16000:
        import torchaudio.transforms as T
        wav = T.Resample(sr, 16000)(wav)
    fb = kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000,
                      frame_shift=10, frame_length=25, dither=0.1)
    return fb.unsqueeze(0), torch.tensor([fb.shape[0]], dtype=torch.long)

fbank_cache = {}
for item in test_items[:sample_n]:
    fbank_cache[item['key']] = extract_fbank(item['wav'])

print(f'Fbank features extracted for {len(fbank_cache)} utterances\n')

# --- Run comparisons ---
all_results = []

for category, variants in COMPARISONS.items():
    print(f'=== Category: {category} ===')
    for variant_name, variant_params in variants.items():
        try:
            total_err, total_len, total_time = 0, 0, 0
            modes = variant_params.get('modes', ['ctc_greedy_search'])
            beam = variant_params.get('beam_size', 1)
            chunk = variant_params.get('chunk_size', -1)
            ctc_w = variant_params.get('ctc_weight', 0.3)

            for item in test_items[:sample_n]:
                fb, fb_len = fbank_cache[item['key']]
                ref = item['txt']

                t0 = time.time()
                with torch.no_grad():
                    results = model.decode(
                        methods=modes, speech=fb, speech_lengths=fb_len,
                        beam_size=beam, decoding_chunk_size=chunk,
                        ctc_weight=ctc_w,
                    )
                total_time += time.time() - t0

                hyp = results[modes[0]][0].tokens
                errors = sum(1 for a,b in zip(hyp, ref) if a!=b) + abs(len(hyp)-len(ref))
                total_err += min(errors, max(len(hyp), len(ref)))
                total_len += len(ref)

            cer = total_err/total_len*100 if total_len>0 else 0
            rtf = total_time / (fb_len.sum().item() * 0.01) if total_len>0 else 0  # 10ms per frame
            latency = chunk * 40 if chunk > 0 else float('inf')  # 40ms per frame after 4x subsampling

            row = {
                'category': category, 'variant': variant_name,
                'CER(%)': f'{cer:.1f}', 'RTF': f'{rtf:.3f}',
                'latency(ms)': f'{latency:.0f}' if latency != float('inf') else 'inf',
                'params': str(variant_params),
            }
            all_results.append(row)
            print(f'  {variant_name}: CER={cer:.1f}%, RTF={rtf:.3f}, latency={row["latency(ms)"]}ms')
        except Exception as e:
            print(f'  {variant_name}: FAILED - {e}')
            all_results.append({
                'category': category, 'variant': variant_name,
                'CER(%)': 'FAIL', 'RTF': 'N/A', 'latency(ms)': 'N/A',
                'params': str(variant_params),
            })

# --- Save results ---
out_dir = Path('results')
out_dir.mkdir(exist_ok=True)

# CSV
with open(out_dir / 'architecture_comparison.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['category','variant','CER(%)','RTF','latency(ms)','params'])
    writer.writeheader()
    writer.writerows(all_results)

# Markdown table
md_lines = ['# Architecture Comparison Results\n']
for cat in ['decode_mode', 'chunk_size', 'ctc_weight']:
    md_lines.append(f'\n## {cat}\n')
    md_lines.append('| Variant | CER(%) | RTF | Latency(ms) | Params |')
    md_lines.append('|---------|--------|-----|------------|--------|')
    for r in all_results:
        if r['category'] == cat:
            md_lines.append(f"| {r['variant']} | {r['CER(%)']} | {r['RTF']} | {r['latency(ms)']} | {r['params']} |")

with open(out_dir / 'architecture_comparison.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f'\nResults saved to results/architecture_comparison.csv + .md')
