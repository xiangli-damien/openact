import argparse
import numpy as np
from pathlib import Path
from openact_core import Run


def main():
    parser = argparse.ArgumentParser(description='Analyze OpenAct run')
    parser.add_argument('run_dir', help='Path to run directory')
    parser.add_argument('--sample', '-s', type=int, default=0, help='Sample index to analyze')
    args = parser.parse_args()

    print('OpenAct Analysis Example')
    print('=' * 60)

    run = Run(args.run_dir)

    print(f'\nRun: {run.run_dir.name}')
    print(f'Model: {run.manifest.model.name}')
    print(f'Task: {run.manifest.dataset.name}')
    print(f'Samples: {len(run)} total, {run.n_valid} valid')

    # run.stats returns a RunStats object with attribute access
    print(f'\nStatistics:')
    print(f'  Total tokens: {run.stats.n_tokens_total:,}')
    print(f'  Success rate: {run.stats.n_samples_ok}/{run.stats.n_samples_total}')

    if run.stats.duration_seconds:
        print(f'  Duration: {run.stats.duration_seconds:.1f}s')

    # ---------- sample inspection ----------
    sample = run[args.sample]

    print(f"\n{'=' * 60}")
    print(f'Sample {args.sample}')
    print(f"{'=' * 60}")

    print(f'\nPrompt (first 200 chars):')
    print(f'  {sample.prompt_text[:200]}...')

    print(f'\nResponse (first 300 chars):')
    print(f'  {sample.response_text[:300]}...')

    print(f'\nToken count: {len(sample.token_ids)}')
    print(f'Hidden states shape: {sample.hidden_states.shape}')

    # ---------- token-text alignment ----------
    print(f"\n{'=' * 60}")
    print('Token-Text Alignment Demo')
    print(f"{'=' * 60}")

    search_text = 'answer'
    span = sample.find_text(search_text)
    if span:
        print(f"\nSearching for: '{search_text}'")
        print(f'  Found at tokens: {span.start}:{span.end}')
        print(f"  Verification: '{sample.span_to_text(span)}'")
        hs = sample.get_span_hidden_states(span, layers=[-1], reduction='mean')
        print(f'  Hidden state shape: {hs.shape}')
        print(f'  Hidden state norm: {np.linalg.norm(hs):.4f}')
    else:
        print(f"\n'{search_text}' not found in response")

    print(f'\nFirst 10 tokens with character spans:')
    aligner = sample.aligner
    for i in range(min(10, len(sample.token_ids))):
        char_span = aligner.token_to_char_span(i)
        text = aligner.token_to_text(i)
        print(f"  Token {i}: chars [{char_span.start}:{char_span.end}] = '{text}'")

    # ---------- text-based hidden state access ----------
    print(f"\n{'=' * 60}")
    print('Text-Based Hidden State Access')
    print(f"{'=' * 60}")

    test_phrases = ['Step', '=', 'answer']
    for phrase in test_phrases:
        try:
            hs = sample.get_text_hidden_states(phrase, layers=[-1], reduction='mean')
            if hs is not None:
                print(f"\n'{phrase}':")
                print(f'  Shape: {hs.shape}')
                print(f'  Norm: {np.linalg.norm(hs):.4f}')
                print(f'  Mean: {np.mean(hs):.6f}')
        except ValueError:
            print(f"\n'{phrase}': not found")

    # ---------- aggregate stats ----------
    print(f"\n{'=' * 60}")
    print('Sample Statistics')
    print(f"{'=' * 60}")

    token_lengths = []
    for s in run.iter_valid():
        token_lengths.append(len(s.token_ids))
        if len(token_lengths) >= 10:
            break

    if token_lengths:
        print(f'\nToken lengths (first {len(token_lengths)} samples):')
        print(f'  Min: {min(token_lengths)}')
        print(f'  Max: {max(token_lengths)}')
        print(f'  Mean: {np.mean(token_lengths):.1f}')

    print(f"\n{'=' * 60}")
    print('Analysis complete!')


if __name__ == '__main__':
    main()