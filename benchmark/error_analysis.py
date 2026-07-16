#!/usr/bin/env python3
"""P0: Error Analysis - Classify ASR errors by type and pattern.
Outputs error_distribution.md with statistics and improvement suggestions."""
import torch, yaml, json, os, sys, soundfile
from collections import Counter, defaultdict
sys.path.insert(0, r'D:\wenet\wenet')
os.chdir(r'D:\wenet\wenet\examples\aishell\s0')

if not hasattr(torch.nn.Module, '__annotations__'):
    torch.nn.Module.__annotations__ = {}

from wenet.utils.init_model import init_model
from wenet.text.char_tokenizer import CharTokenizer
from torchaudio.compliance import kaldi

# Load model
class Args: pass
args = Args(); args.jit = True
args.checkpoint = 'exp/u2pp_conformer_course/epoch_4.pt'
args.config = 'exp/u2pp_conformer_course/train.yaml'
with open(args.config) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
model, configs = init_model(args, configs)
model.eval()
tokenizer = CharTokenizer('data/dict/lang_char.txt', non_lang_syms=None)

# Load test data
test_items = []
with open('data/test/data.list', encoding='utf-8') as f:
    for line in f:
        test_items.append(json.loads(line.strip()))
sample_n = min(50, len(test_items))
print(f'Analyzing {sample_n} utterances...')

# Category-based error analysis: AISHELL has topic labels
# Using first 3 chars of filename to approximate speaker ID grouping
errors_by_type = Counter()  # substitution, deletion, insertion
errors_by_speaker = defaultdict(lambda: {'err': 0, 'total': 0})
errors_by_length = defaultdict(lambda: {'err': 0, 'total': 0})
error_examples = {'substitution': [], 'deletion': [], 'insertion': []}

total_err, total_len = 0, 0

for i, item in enumerate(test_items[:sample_n]):
    try:
        wav, sr = soundfile.read(item['wav'], dtype='float32')
        wav = torch.from_numpy(wav).unsqueeze(0)
        if sr != 16000:
            import torchaudio.transforms as T
            wav = T.Resample(sr, 16000)(wav)
        fb = kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000, frame_shift=10, frame_length=25, dither=0.1)
        fb = fb.unsqueeze(0)
        fb_len = torch.tensor([fb.shape[1]], dtype=torch.long)

        with torch.no_grad():
            results = model.decode(
                methods=['ctc_greedy_search'], speech=fb, speech_lengths=fb_len,
                beam_size=1, decoding_chunk_size=-1)
        hyp = results['ctc_greedy_search'][0].tokens
        ref = item['txt']

        # Detailed alignment via edit distance
        m, n = len(ref), len(hyp)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for a in range(m+1): dp[a][0] = a
        for b in range(n+1): dp[0][b] = b
        for a in range(1, m+1):
            for b in range(1, n+1):
                dp[a][b] = min(dp[a-1][b]+1, dp[a][b-1]+1, dp[a-1][b-1]+(ref[a-1]!=hyp[b-1]))

        # Backtrack
        a, b = m, n
        while a>0 or b>0:
            if a>0 and b>0 and ref[a-1]==hyp[b-1]:
                a-=1; b-=1
            elif a>0 and b>0 and dp[a][b]==dp[a-1][b-1]+1:
                if len(error_examples['substitution'])<3:
                    error_examples['substitution'].append(f"ref='{ref[max(0,a-2):a+2]}' hyp='{hyp[max(0,b-2):b+2]}' => '{ref[a-1]}'->'{hyp[b-1]}'")
                errors_by_type['substitution'] += 1
                a-=1; b-=1
            elif a>0 and dp[a][b]==dp[a-1][b]+1:
                if len(error_examples['deletion'])<3:
                    error_examples['deletion'].append(f"ref='{ref[max(0,a-2):a+2]}' missing '{ref[a-1]}'")
                errors_by_type['deletion'] += 1
                a-=1
            else:
                if len(error_examples['insertion'])<3:
                    error_examples['insertion'].append(f"hyp='{hyp[max(0,b-2):b+2]}' extra '{hyp[b-1]}'")
                errors_by_type['insertion'] += 1
                b-=1

        total_err += dp[m][n]
        total_len += len(ref)

        # Group by ref length
        bucket = len(ref) // 10 * 10
        errors_by_length[bucket]['err'] += dp[m][n]
        errors_by_length[bucket]['total'] += len(ref)

    except Exception as e:
        if i < 3: print(f'  Error {i}: {e}')

# --- Generate report ---
cer = total_err / total_len * 100 if total_len > 0 else 0
total_errors = errors_by_type['substitution'] + errors_by_type['deletion'] + errors_by_type['insertion']

lines = [
    '# Error Analysis Report\n',
    f'## Summary\n',
    f'- Total errors: {total_errors}',
    f'- CER: {cer:.2f}%',
    f'- Test utterances analyzed: {sample_n}',
    '',
    '## Error Type Distribution\n',
    '| Type | Count | Percentage |',
    '|------|-------|------------|',
]
for etype in ['substitution', 'deletion', 'insertion']:
    pct = errors_by_type[etype]/total_errors*100 if total_errors>0 else 0
    lines.append(f'| {etype} | {errors_by_type[etype]} | {pct:.1f}% |')

lines.extend([
    '',
    '## Error Examples\n',
])
for etype in ['substitution', 'deletion', 'insertion']:
    lines.append(f'### {etype}')
    for ex in error_examples[etype]:
        lines.append(f'- {ex}')
    if not error_examples[etype]:
        lines.append('- (no examples collected)')

lines.extend([
    '',
    '## CER by Text Length\n',
    '| Length Range (chars) | CER(%) |',
    '|----------------------|--------|',
])
for bucket in sorted(errors_by_length.keys()):
    d = errors_by_length[bucket]
    bucket_cer = d['err']/d['total']*100 if d['total']>0 else 0
    lines.append(f'| {bucket}-{bucket+9} | {bucket_cer:.1f}% |')

lines.extend([
    '',
    '## Improvement Suggestions\n',
    '1. **高频替换错误** -> 考虑 LM 浅融合(shallow fusion)或引入 N-gram 语言模型',
    '2. **短文本删除错误** -> 可能是 VAD 或静音段处理问题，考虑调整 blank penalty',
    '3. **插入错误** -> 可能噪声被误识别，考虑数据增强(noise augmentation)或 beam search 消歧',
    '4. **长文本 CER 升高** -> 注意力漂移，考虑 MoChA 或 monotonic attention',
    '5. **整体优化** -> 增加训练数据(全量 AISHELL)、更多 epochs、使用预训练模型',
    '',
    '*注：当前模型在 100 utts/5 epochs CPU 上训练，CER 尚未收敛，以上分析展示方法论框架。*',
])

out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'error_analysis.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'Error analysis saved to results/error_analysis.md')
print(f'Total errors: {total_errors}, CER: {cer:.2f}%')
