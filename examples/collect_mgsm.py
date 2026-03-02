"""
Example: Collect MGSM (Multilingual Grade School Math) activations.

Usage:
    python examples/collect_mgsm.py --model Qwen/Qwen2-7B-Instruct --language zh

Collects activation data for the MGSM dataset in a specified language.
Supports: bn, de, en, es, fr, ja, ru, sw, te, th, zh
"""

import argparse
from pathlib import Path

from openact_collect import Collector, ModelManager
from openact_collect.tasks.capability import MGSMTask
from openact_collect.schema import CaptureSpec, GenerationSpec


def main():
    parser = argparse.ArgumentParser(
        description="Collect MGSM activations"
    )
    parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2-7B-Instruct",
    )
    parser.add_argument(
        "--language", "-L", default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=100,
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: runs/mgsm_{language})",
    )
    parser.add_argument(
        "--template", default="cot",
        choices=["cot", "cot_native", "direct", "simple"],
        help="Prompt template variant (default: cot)",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (e.g., '-1,-2,-3,-4')",
    )
    args = parser.parse_args()

    output_dir = args.output or f"runs/mgsm_{args.language}"

    print(f"OpenAct MGSM Collection")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Template: {args.template}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output: {output_dir}")
    print()

    # Create task
    task = MGSMTask(
        max_samples=args.max_samples,
        split="test",
        template=args.template,
        language=args.language,
    )

    # Load model
    model = ModelManager(
        model_name_or_path=args.model,
        dtype="auto",
        device_map="auto",
    )

    # Configure capture
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    capture_spec = CaptureSpec(
        hidden_states=True,
        hidden_states_layers=layers,
        hidden_states_dtype="float16",
    )
    generation_spec = GenerationSpec(
        max_new_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        seed=42,
    )

    # Run collection
    collector = Collector(
        model_manager=model,
        task=task,
        output_dir=output_dir,
        capture_spec=capture_spec,
        generation_spec=generation_spec,
    )

    print("Starting collection...")
    stats = collector.run()

    print()
    print("Collection Complete!")
    print(f"=" * 50)
    print(f"Total processed: {stats['processed']}")
    print(f"Successful: {stats['ok']}")
    print(f"Errors: {stats['error'] + stats['timeout']}")
    print(f"Duration: {stats['duration_seconds']:.1f}s")
    print(f"Output saved to: {output_dir}")

    # Auto-evaluate
    print()
    print("Running evaluation...")
    try:
        from openact_eval import EvalPipeline

        pipeline = EvalPipeline(output_dir)
        summary = pipeline.run_pipeline(progress=True)
        print(f"Accuracy: {summary['accuracy']:.1%}")
        print(f"Correct: {summary['n_correct']}/{summary['n_evaluated']}")
    except ImportError:
        print("Install openact-eval for automatic evaluation.")


if __name__ == "__main__":
    main()
