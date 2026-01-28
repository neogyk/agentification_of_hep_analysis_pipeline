#!/usr/bin/env python3
"""
HEP Analysis Multi-Agent System

Main entry point for the High Energy Physics analysis framework
using smolagents for autonomous analysis workflow.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_model(provider: str = "anthropic", model_id: str = None):
    """
    Initialize the LLM model.

    Args:
        provider: Model provider (anthropic, openai, huggingface)
        model_id: Specific model ID

    Returns:
        Initialized model
    """
    if provider == "anthropic":
        from smolagents import LiteLLMModel
        model_id = model_id or "claude-sonnet-4-20250514"
        return LiteLLMModel(model_id=f"anthropic/{model_id}")

    elif provider == "openai":
        from smolagents import LiteLLMModel
        model_id = model_id or "gpt-4"
        return LiteLLMModel(model_id=model_id)

    elif provider == "huggingface":
        from smolagents import InferenceClientModel
        model_id = model_id or "Qwen/Qwen2.5-72B-Instruct"
        return InferenceClientModel(model_id=model_id)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_interactive_mode(orchestrator):
    """Run interactive command-line interface."""
    print("\n" + "="*60)
    print("HEP Analysis Multi-Agent System")
    print("="*60)
    print("\nAvailable commands:")
    print("  search <query>       - Search physics literature")
    print("  data <source>        - Load data from source")
    print("  code <type>          - Find analysis code")
    print("  select <cuts>        - Apply event selection")
    print("  fit <model>          - Fit distribution")
    print("  significance <n> <b> - Calculate significance")
    print("  workflow <config>    - Run full workflow")
    print("  status               - Show current status")
    print("  quit                 - Exit")
    print("-"*60)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if user_input.lower() == "status":
                status = orchestrator.get_status()
                print(json.dumps(status, indent=2))
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "search":
                result = orchestrator.search_hypothesis(args)
                print(result)

            elif command == "data":
                result = orchestrator.load_data("auto", args)
                print(result)

            elif command == "code":
                result = orchestrator.find_analysis_code(args)
                print(result)

            elif command == "select":
                try:
                    cuts = json.loads(args)
                except:
                    cuts = [{"variable": "pt", "operator": ">", "value": 25}]
                result = orchestrator.select_events(cuts)
                print(result)

            elif command == "fit":
                result = orchestrator.fit_distribution(args or "gaussian", {})
                print(result)

            elif command == "significance":
                parts = args.split()
                observed = int(parts[0]) if parts else 20
                background = float(parts[1]) if len(parts) > 1 else 10
                result = orchestrator.calculate_significance(observed, background)
                print(result)

            elif command == "workflow":
                try:
                    config = json.loads(args) if args else {}
                except:
                    config = {}
                results = orchestrator.run_workflow(config)
                print("\nWorkflow completed!")
                print(f"Completed steps: {results.keys()}")

            else:
                # Run as general task
                result = orchestrator.run(user_input)
                print(result)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_analysis_example(orchestrator):
    """Run example Z boson analysis."""
    print("\n" + "="*60)
    print("Running Example: Z → μμ Cross Section Measurement")
    print("="*60)

    config = {
        "hypothesis": "Standard Model Z boson production in dimuon channel",
        "data_source": "CERN Open Data",
        "channel": "dimuon",
        "selection": {
            "cuts": [
                {"variable": "muon_pt", "operator": ">", "value": 25},
                {"variable": "muon_eta", "operator": "abs<", "value": 2.4},
                {"variable": "muon_iso", "operator": "<", "value": 0.15},
                {"variable": "n_muons", "operator": ">=", "value": 2}
            ]
        },
        "signal_region": {"mass_window": [81, 101]},
        "systematics": [
            "muon_scale",
            "muon_resolution",
            "trigger_efficiency",
            "luminosity",
            "pdf"
        ]
    }

    results = orchestrator.run_workflow(config)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HEP Analysis Multi-Agent System"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "huggingface"],
        default="anthropic",
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (uses default for provider if not specified)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a specific task and exit"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example Z boson analysis"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to analysis configuration JSON file"
    )

    args = parser.parse_args()

    # Initialize model
    print(f"Initializing model (provider: {args.provider})...")
    try:
        model = get_model(args.provider, args.model)
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure you have set the appropriate API key.")
        sys.exit(1)

    # Initialize orchestrator
    from agents import HEPAnalysisOrchestrator
    orchestrator = HEPAnalysisOrchestrator(model)

    # Run based on mode
    if args.task:
        result = orchestrator.run(args.task)
        print(result)

    elif args.example:
        run_analysis_example(orchestrator)

    elif args.config:
        with open(args.config) as f:
            config = json.load(f)
        results = orchestrator.run_workflow(config)
        print("\nAnalysis completed!")
        print(f"Results: {json.dumps(results, indent=2)}")

    elif args.interactive:
        run_interactive_mode(orchestrator)

    else:
        # Default: run interactive mode
        run_interactive_mode(orchestrator)


if __name__ == "__main__":
    main()
