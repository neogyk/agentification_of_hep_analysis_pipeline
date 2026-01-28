# HEP Analysis Multi-Agent System

A comprehensive multi-agent framework built on [smolagents](https://huggingface.co/docs/smolagents) for High Energy Physics (HEP) analysis, designed for particle physics research including new particle searches and hypothesis verification.

## Overview

This system implements a hierarchical multi-agent architecture where specialized agents handle different aspects of a physics analysis:

```
                    ┌─────────────────────────────┐
                    │    HEP Analysis Manager     │
                    │       (Orchestrator)        │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐        ┌─────────────────┐        ┌────────────────┐
│ Infrastructure│        │ Physics Analysis │        │ Interpretation │
│    Agents     │        │     Agents       │        │    Agents      │
└───────────────┘        └─────────────────┘        └────────────────┘
        │                          │                          │
   ┌────┴────┐            ┌────────┴────────┐          ┌──────┴──────┐
   │         │            │                 │          │             │
   ▼         ▼            ▼                 ▼          ▼             ▼
┌─────┐  ┌─────┐    ┌─────────┐      ┌──────────┐  ┌────────┐  ┌────────┐
│ Web │  │Data │    │ Event   │      │ Fitting  │  │Uncert. │  │ Paper  │
│Srch │  │Retr │    │Selection│ ...  │& Stats   │  │Quant.  │  │Writing │
└─────┘  └─────┘    └─────────┘      └──────────┘  └────────┘  └────────┘
```

## Agent Types

### Infrastructure Agents
1. **Web Research Agent** - Searches arXiv, INSPIRE-HEP for papers and measurements
2. **Data Retrieval Agent** - Downloads data from Zenodo, CERN Open Data
3. **Code Search Agent** - Finds analysis code on GitHub/GitLab

### Physics Analysis Agents
4. **Event Selection Agent** - Applies kinematic cuts and selection criteria
5. **Data Processing Agent** - Handles calibrations and derived variables
6. **Reconstruction Agent** - Reconstructs composite particles (Z, W, H)
7. **Fitting Agent** - Fits histograms with signal/background models
8. **Statistical Analysis Agent** - Hypothesis tests, significance, limits
9. **Plotting Agent** - Creates publication-quality plots (CMS/ATLAS style)
10. **Uncertainty Agent** - Calculates systematic/statistical uncertainties
11. **Interpretation Agent** - Interprets results and writes paper sections

## Installation

```bash
# Clone the repository
cd /path/to/agentification_of_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your API keys in a `.env` file:

```env
# Choose one LLM provider
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Optional: for data access
ZENODO_TOKEN=your_zenodo_token
GITHUB_TOKEN=your_github_token
```

## Quick Start

### 1. Interactive Mode

```bash
python main.py --interactive
```

### 2. Run Example Analysis

```bash
python main.py --example
```

### 3. Run Specific Task

```bash
python main.py --task "Search for Z' boson decaying to dimuon pairs"
```

### 4. Python API

```python
from smolagents import LiteLLMModel
from agents import HEPAnalysisOrchestrator

# Initialize model
model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-20250514")

# Create orchestrator
orchestrator = HEPAnalysisOrchestrator(model)

# Run analysis
result = orchestrator.run("Measure Z boson cross section in dimuon channel")

# Or run structured workflow
config = {
    "hypothesis": "Z boson production",
    "channel": "dimuon",
    "selection": {"pt": 25, "eta": 2.4}
}
results = orchestrator.run_workflow(config)
```

## Analysis Workflow

The system follows a typical HEP analysis workflow:

1. **Literature Review** - Search for previous measurements and theory predictions
2. **Data Retrieval** - Download collision data and MC samples
3. **Event Selection** - Apply cuts to select signal events
4. **Reconstruction** - Build physics objects (leptons, jets, MET)
5. **Fitting** - Extract signal yield using fit models
6. **Uncertainties** - Evaluate systematic and statistical errors
7. **Statistical Analysis** - Calculate significance or limits
8. **Interpretation** - Compare with predictions, write results

## Tool Reference

### Physics Tools

| Tool | Description |
|------|-------------|
| `EventSelectionTool` | Apply kinematic cuts and selection |
| `ReconstructionTool` | Reconstruct composite particles |
| `HistogramFitterTool` | Fit distributions with various models |
| `StatisticalFitTool` | Hypothesis tests and limit setting |
| `SystematicUncertaintyTool` | Calculate systematic uncertainties |
| `PublicationPlotTool` | Create CMS/ATLAS style plots |

### Infrastructure Tools

| Tool | Description |
|------|-------------|
| `ArxivSearchTool` | Search arXiv preprints |
| `InspireHEPSearchTool` | Search INSPIRE-HEP database |
| `ZenodoDataTool` | Download data from Zenodo |
| `CERNOpenDataTool` | Access CERN Open Data Portal |
| `ROOTFileLoaderTool` | Load and inspect ROOT files |
| `GitHubSearchTool` | Search for analysis code |

## Examples

### Z Boson Cross Section Measurement

```python
from agents import HEPAnalysisOrchestrator

orchestrator = HEPAnalysisOrchestrator(model)

result = orchestrator.run("""
    Measure the Z boson production cross section in the dimuon channel:
    1. Search for previous CMS/ATLAS measurements
    2. Apply standard dimuon selection (pT > 25 GeV, |η| < 2.4)
    3. Fit the invariant mass spectrum
    4. Calculate cross section with uncertainties
    5. Compare with NNLO prediction
""")
```

### New Particle Search

```python
result = orchestrator.run("""
    Search for a heavy Z' boson decaying to muon pairs:
    1. Define signal region for high-mass dimuon events
    2. Estimate backgrounds from sidebands
    3. Perform bump hunt in mass spectrum
    4. Set upper limits if no excess found
""")
```

## Project Structure

```
agentification_of_analysis/
├── agents/
│   ├── __init__.py
│   ├── infrastructure_agents.py
│   ├── physics_agents.py
│   └── orchestrator.py
├── tools/
│   ├── __init__.py
│   ├── infrastructure.py
│   ├── data_retrieval.py
│   ├── code_tools.py
│   ├── physics_tools.py
│   ├── fitting_tools.py
│   ├── plotting_tools.py
│   ├── uncertainty_tools.py
│   └── interpretation_tools.py
├── examples/
│   ├── __init__.py
│   └── z_boson_analysis.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Supported Models

- **Anthropic Claude** (recommended): `claude-sonnet-4-20250514`, `claude-opus-4-20250514`
- **OpenAI**: `gpt-4`, `gpt-4-turbo`
- **HuggingFace**: `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`

## References

- [smolagents Documentation](https://huggingface.co/docs/smolagents)
- [HuggingFace Agents Course](https://huggingface.co/learn/agents-course)
- [CERN Open Data](http://opendata.cern.ch/)
- [INSPIRE-HEP](https://inspirehep.net/)

## License

MIT License
