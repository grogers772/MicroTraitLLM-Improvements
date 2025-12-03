#!/usr/bin/env python3
"""
Automated model comparison script.

Runs multiple models on the benchmark dataset and collects results for analysis.

Usage:
    python scripts/run_model_comparison.py --models biogpt biomedlm gpt2-baseline
    python scripts/run_model_comparison.py --all --quick  # Quick test with 5 queries
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


AVAILABLE_MODELS = {
    'biogpt': {
        'name': 'BioGPT',
        'description': '347M param biomedical model',
        'config': 'biogpt',
        'supports_8bit': True
    },
    'biomedlm': {
        'name': 'BioMedLM',
        'description': '2.7B param biomedical model',
        'config': 'biomedlm',
        'supports_8bit': True
    },
    'biomedgpt-7b': {
        'name': 'BioMedGPT-7B',
        'description': '7B param biomedical model',
        'config': 'biomedgpt-7b',
        'supports_8bit': True
    },
    'gpt2-baseline': {
        'name': 'GPT-2 Baseline',
        'description': 'General-purpose baseline',
        'config': 'gpt2-baseline',
        'supports_8bit': False
    }
}


def run_single_comparison(
    model: str,
    dataset: str,
    output_dir: str,
    limit: int = None,
    load_8bit: bool = False
) -> Dict[str, Any]:
    """
    Run comparison for a single model.

    Args:
        model: Model identifier
        dataset: Path to dataset
        output_dir: Output directory
        limit: Optional limit on queries
        load_8bit: Whether to use 8-bit loading

    Returns:
        Dictionary with results metadata
    """
    cmd = [
        'python', 'run_comparison.py',
        '--model', model,
        '--dataset', dataset,
        '--output', output_dir
    ]

    if limit:
        cmd.extend(['--limit', str(limit)])

    if load_8bit and AVAILABLE_MODELS[model]['supports_8bit']:
        cmd.append('--load-in-8bit')

    console.print(f"\n[cyan]Running: {' '.join(cmd)}[/cyan]")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.returncode == 0:
            console.print(f"[green]✓ Completed in {duration:.1f}s[/green]")
            return {
                'model': model,
                'status': 'success',
                'duration_seconds': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            console.print(f"[red]✗ Failed with return code {result.returncode}[/red]")
            console.print(f"[red]Error: {result.stderr[:500]}[/red]")
            return {
                'model': model,
                'status': 'failed',
                'duration_seconds': duration,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        console.print(f"[red]✗ Timeout after 1 hour[/red]")
        return {
            'model': model,
            'status': 'timeout',
            'error': 'Execution timeout'
        }

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return {
            'model': model,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run automated model comparison across multiple models'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(AVAILABLE_MODELS.keys()),
        help='Models to compare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all available models'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/samples/microbiological_traits_benchmark_v1.json',
        help='Path to benchmark dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/comparison',
        help='Output directory for results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of queries (for quick testing)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with 5 queries'
    )
    parser.add_argument(
        '--load-8bit',
        action='store_true',
        help='Use 8-bit quantization for supported models'
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.all:
        models_to_run = list(AVAILABLE_MODELS.keys())
    elif args.models:
        models_to_run = args.models
    else:
        console.print("[red]Error: Specify --models or --all[/red]")
        parser.print_help()
        sys.exit(1)

    # Set limit
    limit = args.limit
    if args.quick and not limit:
        limit = 5

    # Print banner
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]MicroTraitLLM - Automated Model Comparison[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    # Print configuration
    table = Table(title="Comparison Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Models", ", ".join(models_to_run))
    table.add_row("Dataset", args.dataset)
    table.add_row("Output", args.output)
    table.add_row("Queries", str(limit) if limit else "All (15)")
    table.add_row("8-bit Loading", "Yes" if args.load_8bit else "No")

    console.print(table)
    console.print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparisons
    results = []

    console.print(f"[bold yellow]Running {len(models_to_run)} model comparisons...[/bold yellow]\n")

    for i, model in enumerate(models_to_run, 1):
        model_info = AVAILABLE_MODELS[model]

        console.print(f"[bold magenta]{'='*80}[/bold magenta]")
        console.print(f"[bold magenta]Model {i}/{len(models_to_run)}: {model_info['name']}[/bold magenta]")
        console.print(f"[bold magenta]{'='*80}[/bold magenta]")
        console.print(f"[dim]{model_info['description']}[/dim]")

        result = run_single_comparison(
            model=model,
            dataset=args.dataset,
            output_dir=str(output_dir),
            limit=limit,
            load_8bit=args.load_8bit
        )

        results.append(result)

    # Save summary
    console.print(f"\n[bold yellow]Saving comparison summary...[/bold yellow]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"comparison_summary_{timestamp}.json"

    summary = {
        'timestamp': timestamp,
        'dataset': args.dataset,
        'limit': limit,
        'load_8bit': args.load_8bit,
        'models': models_to_run,
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    console.print(f"[green]✓ Summary saved to: {summary_file}[/green]")

    # Print final summary
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]Comparison Complete![/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    summary_table = Table(title="Results Summary")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Status", style="yellow")
    summary_table.add_column("Duration", style="magenta")

    for result in results:
        status_color = "green" if result['status'] == 'success' else "red"
        duration = result.get('duration_seconds', 0)
        summary_table.add_row(
            AVAILABLE_MODELS[result['model']]['name'],
            f"[{status_color}]{result['status']}[/{status_color}]",
            f"{duration:.1f}s"
        )

    console.print(summary_table)

    # Next steps
    console.print(f"\n[bold green]Next Steps:[/bold green]")
    console.print(f"1. Analyze results: [cyan]python scripts/analyze_results.py --results {output_dir}[/cyan]")
    console.print(f"2. View summary: [cyan]cat {summary_file}[/cyan]")
    console.print(f"3. Generate report: [cyan]python scripts/generate_report.py --results {output_dir}[/cyan]\n")

    # Exit code based on results
    failed = sum(1 for r in results if r['status'] != 'success')
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
