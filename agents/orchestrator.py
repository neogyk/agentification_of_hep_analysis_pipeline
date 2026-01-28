"""
HEP Analysis Orchestrator

Main orchestrator that coordinates all specialized agents for
complete HEP analysis workflow.
"""

from typing import Optional, List, Dict, Any, Union
from smolagents import CodeAgent, ToolCallingAgent
from dataclasses import dataclass, field
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .infrastructure_agents import (
    create_web_research_agent,
    create_data_retrieval_agent,
    create_code_search_agent,
)
from .physics_agents import (
    create_event_selection_agent,
    create_data_processing_agent,
    create_reconstruction_agent,
    create_fitting_agent,
    create_statistical_analysis_agent,
    create_plotting_agent,
    create_uncertainty_agent,
    create_interpretation_agent,
)


@dataclass
class AnalysisState:
    """Track the state of the analysis workflow."""
    current_step: str = "initialized"
    completed_steps: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    data_loaded: bool = False
    events_selected: bool = False
    fit_performed: bool = False


class HEPAnalysisOrchestrator:
    """
    Main orchestrator for HEP analysis multi-agent system.

    Coordinates specialized agents to perform complete physics analysis:
    1. Literature research and hypothesis definition
    2. Data retrieval and loading
    3. Event selection and reconstruction
    4. Fitting and statistical analysis
    5. Uncertainty quantification
    6. Results interpretation and paper writing
    """

    def __init__(self, model, config: Optional[Dict] = None):
        """
        Initialize the HEP analysis orchestrator.

        Args:
            model: LLM model to use for all agents
            config: Optional configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.state = AnalysisState()

        # Initialize all specialized agents
        self._initialize_agents()

        # Create manager agent that coordinates sub-agents
        self._create_manager_agent()

    def _initialize_agents(self):
        """Initialize all specialized agents."""
        print("Initializing HEP Analysis Agents...")

        # Infrastructure agents
        self.web_research_agent = create_web_research_agent(self.model)
        self.data_retrieval_agent = create_data_retrieval_agent(self.model)
        self.code_search_agent = create_code_search_agent(self.model)

        # Physics analysis agents
        self.event_selection_agent = create_event_selection_agent(self.model)
        self.data_processing_agent = create_data_processing_agent(self.model)
        self.reconstruction_agent = create_reconstruction_agent(self.model)
        self.fitting_agent = create_fitting_agent(self.model)
        self.statistical_agent = create_statistical_analysis_agent(self.model)
        self.plotting_agent = create_plotting_agent(self.model)
        self.uncertainty_agent = create_uncertainty_agent(self.model)
        self.interpretation_agent = create_interpretation_agent(self.model)

        print("All agents initialized successfully!")

    def _create_manager_agent(self):
        """Create the manager agent that coordinates all sub-agents."""
        from smolagents import ManagedAgent

        # Wrap agents as managed agents
        managed_agents = [
            ManagedAgent(
                agent=self.web_research_agent,
                name="web_research",
                description="Searches physics literature, arXiv, and INSPIRE-HEP. Use for finding papers, measurements, and theoretical predictions."
            ),
            ManagedAgent(
                agent=self.data_retrieval_agent,
                name="data_retrieval",
                description="Downloads and loads physics data from Zenodo, CERN Open Data, and ROOT files."
            ),
            ManagedAgent(
                agent=self.code_search_agent,
                name="code_search",
                description="Searches GitHub/GitLab for analysis code, analyzes code structure, generates code templates."
            ),
            ManagedAgent(
                agent=self.event_selection_agent,
                name="event_selection",
                description="Applies event selection cuts, generates cut flows, calculates selection efficiencies."
            ),
            ManagedAgent(
                agent=self.data_processing_agent,
                name="data_processing",
                description="Processes data: applies calibrations, corrections, calculates derived variables."
            ),
            ManagedAgent(
                agent=self.reconstruction_agent,
                name="reconstruction",
                description="Reconstructs physics objects (Z, W, H candidates), calculates invariant masses."
            ),
            ManagedAgent(
                agent=self.fitting_agent,
                name="fitting",
                description="Fits histograms with signal/background models, estimates backgrounds, extracts signal yields."
            ),
            ManagedAgent(
                agent=self.statistical_agent,
                name="statistical_analysis",
                description="Performs hypothesis tests, calculates significance, sets upper limits."
            ),
            ManagedAgent(
                agent=self.plotting_agent,
                name="plotting",
                description="Creates publication-quality plots with CMS/ATLAS styling."
            ),
            ManagedAgent(
                agent=self.uncertainty_agent,
                name="uncertainty",
                description="Calculates systematic and statistical uncertainties, builds covariance matrices."
            ),
            ManagedAgent(
                agent=self.interpretation_agent,
                name="interpretation",
                description="Interprets results, compares with SM predictions, writes paper sections."
            ),
        ]

        # Create manager agent
        self.manager = CodeAgent(
            tools=[],
            model=self.model,
            managed_agents=managed_agents,
            max_steps=50,
            planning_interval=5,
            additional_authorized_imports=[
                "numpy", "json", "pathlib", "os"
            ]
        )

    def run(self, task: str, **kwargs) -> str:
        """
        Run a complete analysis task.

        Args:
            task: Description of the analysis task to perform
            **kwargs: Additional arguments

        Returns:
            Analysis results as string
        """
        print(f"\n{'='*60}")
        print("HEP ANALYSIS ORCHESTRATOR")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        self.state.current_step = "running"

        try:
            result = self.manager.run(task, **kwargs)
            self.state.current_step = "completed"
            self.state.results["final"] = result
            return result

        except Exception as e:
            self.state.current_step = "error"
            self.state.errors.append(str(e))
            return f"Analysis error: {str(e)}"

    def run_workflow(self, analysis_config: Dict) -> Dict:
        """
        Run a structured analysis workflow.

        Args:
            analysis_config: Configuration dictionary with:
                - hypothesis: Physics hypothesis to test
                - data_source: Where to get data
                - channel: Analysis channel (e.g., 'dimuon')
                - selection: Event selection criteria
                - signal_region: Signal region definition
                - systematics: List of systematic sources

        Returns:
            Dictionary with all analysis results
        """
        results = {}

        # Step 1: Literature Research
        print("\n[Step 1/8] Literature Research...")
        hypothesis = analysis_config.get("hypothesis", "Standard Model Z boson production")
        research_result = self.web_research_agent.run(
            f"Search for previous measurements and theoretical predictions for: {hypothesis}"
        )
        results["literature_review"] = research_result
        self.state.completed_steps.append("literature_review")

        # Step 2: Data Retrieval
        print("\n[Step 2/8] Data Retrieval...")
        data_source = analysis_config.get("data_source", "CERN Open Data")
        data_result = self.data_retrieval_agent.run(
            f"Find and load data from {data_source} for {analysis_config.get('channel', 'dimuon')} analysis"
        )
        results["data_retrieval"] = data_result
        self.state.data_loaded = True
        self.state.completed_steps.append("data_retrieval")

        # Step 3: Event Selection
        print("\n[Step 3/8] Event Selection...")
        selection = analysis_config.get("selection", {"pt": 25, "eta": 2.4})
        selection_result = self.event_selection_agent.run(
            f"Apply event selection with criteria: {json.dumps(selection)}"
        )
        results["event_selection"] = selection_result
        self.state.events_selected = True
        self.state.completed_steps.append("event_selection")

        # Step 4: Reconstruction
        print("\n[Step 4/8] Reconstruction...")
        channel = analysis_config.get("channel", "dimuon")
        reco_result = self.reconstruction_agent.run(
            f"Reconstruct {channel} candidates and calculate invariant mass"
        )
        results["reconstruction"] = reco_result
        self.state.completed_steps.append("reconstruction")

        # Step 5: Fitting
        print("\n[Step 5/8] Fitting...")
        fit_result = self.fitting_agent.run(
            f"Fit the invariant mass distribution with signal+background model"
        )
        results["fitting"] = fit_result
        self.state.fit_performed = True
        self.state.completed_steps.append("fitting")

        # Step 6: Uncertainty Quantification
        print("\n[Step 6/8] Uncertainty Quantification...")
        systematics = analysis_config.get("systematics", ["luminosity", "efficiency"])
        uncertainty_result = self.uncertainty_agent.run(
            f"Calculate systematic uncertainties for: {systematics}"
        )
        results["uncertainties"] = uncertainty_result
        self.state.completed_steps.append("uncertainties")

        # Step 7: Statistical Analysis
        print("\n[Step 7/8] Statistical Analysis...")
        stat_result = self.statistical_agent.run(
            "Perform statistical analysis: calculate significance and compatibility with SM"
        )
        results["statistical_analysis"] = stat_result
        self.state.completed_steps.append("statistical_analysis")

        # Step 8: Interpretation and Documentation
        print("\n[Step 8/8] Results Interpretation...")
        interp_result = self.interpretation_agent.run(
            f"Interpret results for {hypothesis} and generate paper abstract"
        )
        results["interpretation"] = interp_result
        self.state.completed_steps.append("interpretation")

        # Generate plots
        print("\n[Plotting] Creating figures...")
        plot_result = self.plotting_agent.run(
            "Create publication-quality plots for the analysis results"
        )
        results["plots"] = plot_result
        self.state.completed_steps.append("plotting")

        self.state.current_step = "workflow_completed"
        self.state.results = results

        return results

    def get_status(self) -> Dict:
        """Get current analysis status."""
        return {
            "current_step": self.state.current_step,
            "completed_steps": self.state.completed_steps,
            "data_loaded": self.state.data_loaded,
            "events_selected": self.state.events_selected,
            "fit_performed": self.state.fit_performed,
            "errors": self.state.errors
        }

    def search_hypothesis(self, hypothesis: str) -> str:
        """Search for information about a physics hypothesis."""
        return self.web_research_agent.run(
            f"Search for papers and measurements related to: {hypothesis}"
        )

    def load_data(self, source: str, dataset: str) -> str:
        """Load data from specified source."""
        return self.data_retrieval_agent.run(
            f"Load data from {source}, dataset: {dataset}"
        )

    def find_analysis_code(self, analysis_type: str) -> str:
        """Find example code for a type of analysis."""
        return self.code_search_agent.run(
            f"Find example code for {analysis_type} analysis"
        )

    def select_events(self, cuts: List[Dict]) -> str:
        """Apply event selection cuts."""
        return self.event_selection_agent.run(
            f"Apply event selection: {json.dumps(cuts)}"
        )

    def reconstruct_objects(self, object_type: str) -> str:
        """Reconstruct physics objects."""
        return self.reconstruction_agent.run(
            f"Reconstruct {object_type} candidates"
        )

    def fit_distribution(self, model: str, config: Dict) -> str:
        """Fit a distribution with specified model."""
        return self.fitting_agent.run(
            f"Fit with {model} model, config: {json.dumps(config)}"
        )

    def calculate_uncertainties(self, sources: List[str]) -> str:
        """Calculate systematic uncertainties."""
        return self.uncertainty_agent.run(
            f"Calculate uncertainties for sources: {sources}"
        )

    def calculate_significance(self, observed: int, background: float) -> str:
        """Calculate discovery significance."""
        return self.statistical_agent.run(
            f"Calculate significance: observed={observed}, background={background}"
        )

    def create_plots(self, plot_type: str, data: Dict) -> str:
        """Create analysis plots."""
        return self.plotting_agent.run(
            f"Create {plot_type} plot with data: {json.dumps(data)}"
        )

    def interpret_results(self, results: Dict) -> str:
        """Interpret analysis results."""
        return self.interpretation_agent.run(
            f"Interpret results: {json.dumps(results)}"
        )

    def write_paper_section(self, section: str, info: Dict) -> str:
        """Generate a paper section."""
        return self.interpretation_agent.run(
            f"Write {section} section for paper with info: {json.dumps(info)}"
        )
