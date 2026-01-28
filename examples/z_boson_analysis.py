#!/usr/bin/env python3
"""
Example: Z Boson Cross Section Measurement

This example demonstrates how to use the HEP Analysis Multi-Agent System
to perform a complete Z → μμ cross section measurement.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import LiteLLMModel, InferenceClientModel


def main():
    """Run Z boson analysis example."""

    # Initialize model (choose one based on your API keys)
    # Option 1: Anthropic Claude
    # model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-20250514")

    # Option 2: OpenAI
    # model = LiteLLMModel(model_id="gpt-4")

    # Option 3: HuggingFace (free)
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

    # Import and create orchestrator
    from agents import HEPAnalysisOrchestrator
    orchestrator = HEPAnalysisOrchestrator(model)

    # Define analysis configuration
    analysis_config = {
        "hypothesis": "Z boson production cross section in pp collisions at 13 TeV",
        "data_source": "CERN Open Data CMS Run 2",
        "channel": "dimuon",
        "sqrt_s": 13,  # TeV
        "luminosity": 139,  # fb^-1

        "selection": {
            "cuts": [
                {"variable": "muon_pt", "operator": ">", "value": 25, "description": "Leading muon pT > 25 GeV"},
                {"variable": "muon_eta", "operator": "abs<", "value": 2.4, "description": "|η| < 2.4"},
                {"variable": "muon_iso", "operator": "<", "value": 0.15, "description": "Relative isolation < 0.15"},
                {"variable": "n_muons", "operator": ">=", "value": 2, "description": "At least 2 muons"},
                {"variable": "dimuon_mass", "operator": "between", "value": [60, 120], "description": "60 < m_μμ < 120 GeV"}
            ]
        },

        "signal_region": {
            "mass_window": [81, 101],
            "description": "Z mass window ±10 GeV"
        },

        "fit_config": {
            "signal_model": "breit_wigner_convolved_gaussian",
            "background_model": "exponential",
            "fit_range": [60, 120]
        },

        "systematics": [
            "muon_momentum_scale",
            "muon_momentum_resolution",
            "muon_identification",
            "muon_isolation",
            "trigger_efficiency",
            "pileup",
            "luminosity",
            "pdf",
            "theory_scale"
        ],

        "output": {
            "directory": "./results/z_cross_section",
            "formats": ["pdf", "png", "root"]
        }
    }

    print("="*70)
    print("Z Boson Cross Section Measurement")
    print("="*70)
    print(f"\nChannel: {analysis_config['channel']}")
    print(f"√s = {analysis_config['sqrt_s']} TeV")
    print(f"Luminosity: {analysis_config['luminosity']} fb⁻¹")
    print("="*70)

    # Run the full analysis workflow
    results = orchestrator.run_workflow(analysis_config)

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    for step, result in results.items():
        print(f"\n[{step}]")
        if isinstance(result, str):
            # Print first 500 chars
            print(result[:500] + "..." if len(result) > 500 else result)
        else:
            print(result)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

    return results


def run_individual_agents_example():
    """Example of using individual agents."""

    from smolagents import InferenceClientModel
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

    # Import individual agents
    from agents import (
        WebResearchAgent,
        DataRetrievalAgent,
        EventSelectionAgent,
        FittingAgent,
        UncertaintyAgent,
        InterpretationAgent
    )

    print("\n" + "="*70)
    print("Individual Agent Examples")
    print("="*70)

    # 1. Web Research Agent
    print("\n[1] Web Research Agent")
    web_agent = WebResearchAgent(model)
    result = web_agent.run(
        "Find the latest CMS measurement of Z boson cross section at 13 TeV"
    )
    print(result[:500])

    # 2. Event Selection Agent
    print("\n[2] Event Selection Agent")
    selection_agent = EventSelectionAgent(model)
    cuts = [
        {"variable": "pt", "operator": ">", "value": 25},
        {"variable": "eta", "operator": "abs<", "value": 2.4},
        {"variable": "mass", "operator": "between", "value": [81, 101]}
    ]
    result = selection_agent.run(
        f"Apply event selection with these cuts: {cuts}"
    )
    print(result[:500])

    # 3. Fitting Agent
    print("\n[3] Fitting Agent")
    fitting_agent = FittingAgent(model)
    result = fitting_agent.run(
        "Fit the Z boson mass peak with a Breit-Wigner signal and exponential background"
    )
    print(result[:500])

    # 4. Uncertainty Agent
    print("\n[4] Uncertainty Agent")
    uncertainty_agent = UncertaintyAgent(model)
    result = uncertainty_agent.run(
        "Calculate systematic uncertainty from muon momentum scale with 1% variation"
    )
    print(result[:500])

    # 5. Interpretation Agent
    print("\n[5] Interpretation Agent")
    interp_agent = InterpretationAgent(model)
    result = interp_agent.run(
        "Interpret a measured cross section of 1.95 nb with 3% uncertainty compared to SM prediction of 1.92 nb"
    )
    print(result[:500])


def run_new_particle_search_example():
    """Example: Search for a new particle."""

    from smolagents import InferenceClientModel
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

    from agents import HEPAnalysisOrchestrator
    orchestrator = HEPAnalysisOrchestrator(model)

    print("\n" + "="*70)
    print("New Particle Search Example")
    print("="*70)

    # High-level task description
    task = """
    Perform a search for a new heavy neutral resonance (Z') decaying to dimuon pairs.

    Steps to follow:
    1. Search literature for current Z' limits and theoretical predictions
    2. Define event selection optimized for high-mass dimuon events
    3. Estimate backgrounds using data-driven methods
    4. Fit the dimuon mass spectrum with signal + background model
    5. Calculate expected and observed upper limits on Z' production
    6. Generate publication-quality plots
    7. Interpret results and write summary
    """

    result = orchestrator.run(task)
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=["full", "individual", "search"],
                       default="full", help="Which example to run")
    args = parser.parse_args()

    if args.example == "full":
        main()
    elif args.example == "individual":
        run_individual_agents_example()
    elif args.example == "search":
        run_new_particle_search_example()
