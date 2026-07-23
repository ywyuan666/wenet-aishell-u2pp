#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CER evaluation for WeNet AISHELL model with proper fbank feature extraction.

Usage:
    python eval_cer.py                            # default: first 20 test utts
    python eval_cer.py --subset 100               # first 100 test utts
    python eval_cer.py --subset -1                # all test utts
"""
import torch, yaml, json, os, sys, soundfile, argparse

WENET_DIR = os.environ.get('WENET_DIR', r'D:\wenet\wenet')
S0_DIR = os.environ.get('WENET_S0_DIR', r'D:\wenet\wenet\examples\aishell\s0')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=20, help='-1 for all')
    args = parser.parse_args()

    sys.path.insert(0, WENET_DIR)
    os.chdir(S0_DIR)

    if not hasattr(torch.nn.Module, '__annotations__'):
        torch.nn.Module.__annotations__ = {}

    from wenet.utils.init_model import init_model
    from wenet.text.char_tokenizer import CharTokenizer
    from torchaudio.compliance import kaldi

    # Load model
    with open('exp/u2pp_conformer_course/train.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    class Args: pass
    a = Args(); a.jit = True
    a.checkpoint = 'exp/u2pp_conformer_course/epoch_4.pt'
    a.config = 'exp/u2pp_conformer_course/train.yaml'
    model, configs = init_model(a, configs)
    model.eval()

    tokenizer = CharTokenizer('data/dict/lang_char.txt', non_lang_syms=None)

    # Load test data
    test_items = []
    with open('data/test/data.list', encoding='utf-8') as f:
        for line in f:
            test_items.append(json.loads(line.strip()))
    print(f'Test set: {len(test_items)} utterances')

    sample_n = len(test_items) if args.subset < 0 else min(args.subset, len(test_items))
    total_err = 0
    total_len = 0
    decoded = []

    for i, item in enumerate(test_items[:sample_n]):
        try:
            wav, sr = soundfile.read(item['wav'], dtype='float32')
            wav = torch.from_numpy(wav).unsqueeze(0)
            if sr != 16000:
                import torchaudio.transforms as T
                wav = T.Resample(sr, 16000)(wav)
            fbank = kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000,
                               frame_shift=10, frame_length=25, dither=0.1)
            fbank = fbank.unsqueeze(0)
            fbank_len = torch.tensor([fbank.shape[1]], dtype=torch.long)

            with torch.no_grad():
                results = model.decode(
                    methods=['ctc_greedy_search'],
                    speech=fbank,
                    speech_lengths=fbank_len,
                    beam_size=1,
                    decoding_chunk_size=-1
                )
            hyp_text = results['ctc_greedy_search'][0].tokens
            ref_text = item['txt']

            errors = sum(1 for a, b in zip(hyp_text, ref_text) if a != b)
            errors += abs(len(hyp_text) - len(ref_text))
            total_err += min(errors, max(len(hyp_text), len(ref_text)))
            total_len += len(ref_text)
            decoded.append(f'{item["key"]}\t{hyp_text}')

            if i % 5 == 0:
                print(f'  [{i}/{sample_n}] ref={ref_text[:15]}... hyp={hyp_text[:15]}...', flush=True)
        except Exception as e:
            print(f'  Error {i}: {e}', flush=True)

    cer = total_err / total_len * 100 if total_len > 0 else 0
    print(f'\n=== CER Result ===')
    print(f'Test samples: {sample_n}')
    print(f'CER: {cer:.2f}%')
    print(f'Errors: {total_err}, Total chars: {total_len}')

    out_dir = 'exp/u2pp_conformer_course/dec_test/ctc_greedy_search'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'text'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(decoded) + '\n')
    with open(os.path.join(out_dir, 'cer_result.txt'), 'w', encoding='utf-8') as f:
        f.write(f'CER on {sample_n} of {len(test_items)} test utterances: {cer:.2f}%\n')
        f.write(f'Errors: {total_err}, Total chars: {total_len}\n')
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()
