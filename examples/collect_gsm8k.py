import argparse
from pathlib import Path
from openact_collect import Collector, ModelManager
from openact_collect.tasks.capability import GSM8KTask
from openact_collect.schema import CaptureSpec, GenerationSpec


def main():
    parser = argparse.ArgumentParser(description='Collect GSM8K activations')
    parser.add_argument('--model', '-m', default='Qwen/Qwen2-7B-Instruct')
    parser.add_argument('--max-samples', '-n', type=int, default=100)
    parser.add_argument('--output', '-o', default='runs/gsm8k_example')
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g., '-1,-2,-3')",
    )
    args = parser.parse_args()

    print(f'OpenAct Collection Example')
    print(f'=' * 50)
    print(f'Model: {args.model}')
    print(f'Max samples: {args.max_samples}')
    print(f'Output: {args.output}')
    print()

    task = GSM8KTask(max_samples=args.max_samples, split='test', template='cot')

    # NOTE: The parameter is model_name_or_path, not model_name
    model = ModelManager(model_name_or_path=args.model, dtype='auto', device_map='auto')

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]

    capture_spec = CaptureSpec(
        hidden_states=True,
        hidden_states_layers=layers,
        hidden_states_dtype='float16',
    )
    generation_spec = GenerationSpec(
        max_new_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        seed=42,
    )

    collector = Collector(
        model_manager=model,
        task=task,
        output_dir=args.output,
        capture_spec=capture_spec,
        generation_spec=generation_spec,
    )

    print('Starting collection...')
    stats = collector.run()

    print()
    print('Collection Complete!')
    print(f'=' * 50)
    print(f"Total processed: {stats['processed']}")
    print(f"Successful: {stats['ok']}")
    print(f"Errors: {stats['error'] + stats['timeout']}")
    print(f"Duration: {stats['duration_seconds']:.1f}s")
    print(f'Output saved to: {args.output}')


if __name__ == '__main__':
    main()