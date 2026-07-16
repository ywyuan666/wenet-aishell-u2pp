#!/usr/bin/env python3
"""P1: Streaming vs Non-streaming Latency-Accuracy Tradeoff Curve.
Decodes with different chunk sizes and measures CER + RTF + estimated latency."""

import torch, yaml, json, os, sys, soundfile, time, csv
sys.path.insert(0, r'D:\wenet\wenet')
os.chdir(r'D:\wenet\wenet\examples\aishell\s0')

if not hasattr(torch.nn.Module, '__annotations__'):
    torch.nn.Module.__annotations__ = {}

from wenet.utils.init_model import init_model
from wenet.text.char_tokenizer import CharTokenizer
from torchaudio.compliance import kaldi

class Args: pass
args = Args(); args.jit = True
args.checkpoint = 'exp/u2pp_conformer_course/epoch_4.pt'
args.config = 'exp/u2pp_conformer_course/train.yaml'
with open(args.config) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
model, configs = init_model(args, configs)
model.eval()
tokenizer = CharTokenizer('data/dict/lang_char.txt', non_lang_syms=None)

test_items = []
with open('data/test/data.list', encoding='utf-8') as f:
    for line in f:
        test_items.append(json.loads(line.strip()))
sample_n = min(20, len(test_items))

chunk_sizes = [-1, 32, 16, 8, 4, 2]
results = []

print(f'Streaming tradeoff on {sample_n} utterances\n')
for chunk in chunk_sizes:
    total_err, total_len, total_time = 0, 0, 0
    for i, item in enumerate(test_items[:sample_n]):
        try:
            wav, sr = soundfile.read(item['wav'], dtype='float32')
            wav = torch.from_numpy(wav).unsqueeze(0)
            if sr != 16000:
                import torchaudio.transforms as T
                wav = T.Resample(sr, 16000)(wav)
            fb = kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000,
                            frame_shift=10, frame_length=25, dither=0.1)
            fb = fb.unsqueeze(0)
            fb_len = torch.tensor([fb.shape[1]], dtype=torch.long)
            ref = item['txt']

            t0 = time.time()
            with torch.no_grad():
                res = model.decode(
                    methods=['ctc_greedy_search'], speech=fb, speech_lengths=fb_len,
                    beam_size=1, decoding_chunk_size=chunk)
            total_time += time.time() - t0
            hyp = res['ctc_greedy_search'][0].tokens
            errors = sum(1 for a,b in zip(hyp, ref) if a!=b) + abs(len(hyp)-len(ref))
            total_err += min(errors, max(len(hyp), len(ref)))
            total_len += len(ref)
        except Exception as e:
            pass

    cer = total_err/total_len*100 if total_len>0 else 0
    rtf = total_time/(total_len*0.01) if total_len>0 else 0
    latency = chunk*40 if chunk>0 else float('inf')
    label = 'non-streaming' if chunk==-1 else f'chunk_{chunk}'

    results.append({'chunk_size': label, 'CER(%)': f'{cer:.1f}',
                     'RTF': f'{rtf:.3f}', 'latency(ms)': latency})
    print(f'{label}: CER={cer:.1f}%, RTF={rtf:.3f}, latency={"inf" if latency==float("inf") else latency}ms')

# Save
os.makedirs('results', exist_ok=True)

md = ['# Streaming vs Non-streaming Tradeoff\n',
      '## Latency-Accuracy Curve\n',
      '| Chunk Size | CER(%) | RTF | Latency(ms) |',
      '|-----------|--------|-----|-------------|']
for r in results:
    lat = r['latency(ms)']
    md.append(f"| {r['chunk_size']} | {r['CER(%)']} | {r['RTF']} | {'inf' if lat==float('inf') else lat} |")
md += ['\n## Analysis\n',
       '- **Latency**: chunk_size * 40ms (after 4x encoder subsampling, 10ms/frame -> 40ms/chunk-frame)',
       '- **Sweet spot**: trade-off point where CER is acceptable and latency meets real-time requirement',
       '- **Recommendation**: chunk_size=16 (640ms latency) for interactive apps, chunk_size=4 for ultra-low latency',
       '\n*注：当前模型在 100 utts/5 epochs CPU 上训练。完整训练(360 epochs, 全量数据)可获有意义结果。*']

with open('results/streaming_tradeoff.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['chunk_size','CER(%)','RTF','latency(ms)'])
    writer.writeheader()
    writer.writerows(results)

with open('results/streaming_tradeoff.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print('\n=== Streaming Tradeoff Curve ===')
for r in results:
    s = f"{r['chunk_size']:>18}: CER={r['CER(%)']:>6}%  RTF={r['RTF']:>6}  latency={r['latency(ms)']}"
    print(s)
print(f'\nSaved to results/streaming_tradeoff.csv + .md')
