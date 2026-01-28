"""
Physics Analysis Agents for HEP Analysis

Specialized agents for each step of HEP analysis workflow.
"""

from typing import Optional, List, Dict, Any
from smolagents import CodeAgent, ToolCallingAgent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.physics_tools import (
    EventSelectionTool,
    DataProcessingTool,
    ReconstructionTool,
    KinematicCalculatorTool,
)
from tools.fitting_tools import (
    HistogramFitterTool,
    BackgroundModelTool,
    SignalModelTool,
    StatisticalFitTool,
)
from tools.plotting_tools import (
    HistogramPlotTool,
    FitResultPlotTool,
    ComparisonPlotTool,
    PublicationPlotTool,
)
from tools.uncertainty_tools import (
    SystematicUncertaintyTool,
    StatisticalUncertaintyTool,
    UncertaintyCombinerTool,
    CovarianceMatrixTool,
)
from tools.interpretation_tools import (
    ResultsInterpreterTool,
    SignificanceCalculatorTool,
    LimitCalculatorTool,
    PaperWriterTool,
)


def create_event_selection_agent(model, name: str = "event_selection_agent"):
    """Create agent for event selection and filtering."""
    tools = [
        EventSelectionTool(),
        DataProcessingTool(),
        KinematicCalculatorTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Event selection agent for HEP analysis.
        Applies kinematic cuts and selection criteria to physics data.
        Generates cut flow tables and calculates selection efficiencies.
        Use this agent to filter events and define signal/control regions.""",
        additional_authorized_imports=[
            "numpy", "json", "awkward", "uproot"
        ]
    )


def create_data_processing_agent(model, name: str = "data_processing_agent"):
    """Create agent for data processing and transformation."""
    tools = [
        DataProcessingTool(),
        KinematicCalculatorTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Data processing agent for HEP analysis.
        Applies calibrations, corrections, and transformations to data.
        Calculates derived kinematic variables.
        Use this agent to prepare data for analysis.""",
        additional_authorized_imports=[
            "numpy", "pandas", "json", "scipy"
        ]
    )


def create_reconstruction_agent(model, name: str = "reconstruction_agent"):
    """Create agent for physics object reconstruction."""
    tools = [
        ReconstructionTool(),
        KinematicCalculatorTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Reconstruction agent for particle physics.
        Reconstructs composite objects (Z, W, H candidates).
        Calculates invariant masses and kinematic variables.
        Use this agent to build physics objects from detector-level data.""",
        additional_authorized_imports=[
            "numpy", "vector", "awkward", "json"
        ]
    )


def create_fitting_agent(model, name: str = "fitting_agent"):
    """Create agent for histogram fitting and signal extraction."""
    tools = [
        HistogramFitterTool(),
        BackgroundModelTool(),
        SignalModelTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=20,
        name=name,
        description="""Fitting agent for HEP analysis.
        Fits histograms with signal and background models.
        Estimates backgrounds using data-driven methods.
        Extracts signal yields and fit parameters.
        Use this agent for signal extraction and background estimation.""",
        additional_authorized_imports=[
            "numpy", "scipy", "iminuit", "hist", "json"
        ]
    )


def create_statistical_analysis_agent(model, name: str = "statistical_agent"):
    """Create agent for statistical analysis and hypothesis testing."""
    tools = [
        StatisticalFitTool(),
        SignificanceCalculatorTool(),
        LimitCalculatorTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=20,
        name=name,
        description="""Statistical analysis agent for HEP.
        Performs profile likelihood fits and hypothesis tests.
        Calculates discovery significance and p-values.
        Sets upper limits using CLs and Bayesian methods.
        Use this agent for statistical interpretation of results.""",
        additional_authorized_imports=[
            "numpy", "scipy", "json", "pyhf"
        ]
    )


def create_plotting_agent(model, name: str = "plotting_agent"):
    """Create agent for creating physics plots."""
    tools = [
        HistogramPlotTool(),
        FitResultPlotTool(),
        ComparisonPlotTool(),
        PublicationPlotTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Plotting agent for HEP analysis.
        Creates publication-quality plots with CMS/ATLAS styling.
        Generates data/MC comparisons, fit results, and ratio panels.
        Use this agent to create figures for papers and presentations.""",
        additional_authorized_imports=[
            "numpy", "matplotlib", "mplhep", "json", "pathlib"
        ]
    )


def create_uncertainty_agent(model, name: str = "uncertainty_agent"):
    """Create agent for uncertainty quantification."""
    tools = [
        SystematicUncertaintyTool(),
        StatisticalUncertaintyTool(),
        UncertaintyCombinerTool(),
        CovarianceMatrixTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=20,
        name=name,
        description="""Uncertainty quantification agent for HEP.
        Calculates systematic and statistical uncertainties.
        Combines uncertainties accounting for correlations.
        Builds and analyzes covariance matrices.
        Use this agent for complete uncertainty evaluation.""",
        additional_authorized_imports=[
            "numpy", "scipy", "json"
        ]
    )


def create_interpretation_agent(model, name: str = "interpretation_agent"):
    """Create agent for results interpretation and documentation."""
    tools = [
        ResultsInterpreterTool(),
        SignificanceCalculatorTool(),
        PaperWriterTool(),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Results interpretation agent for HEP.
        Interprets physics results in context of Standard Model.
        Compares measurements with predictions and previous results.
        Generates paper sections and documentation.
        Use this agent for final interpretation and paper writing.""",
        additional_authorized_imports=[
            "numpy", "json", "pathlib"
        ]
    )


# Convenience class wrappers

class EventSelectionAgent:
    """Event selection and filtering agent."""

    def __init__(self, model):
        self.agent = create_event_selection_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class DataProcessingAgent:
    """Data processing and transformation agent."""

    def __init__(self, model):
        self.agent = create_data_processing_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class ReconstructionAgent:
    """Physics object reconstruction agent."""

    def __init__(self, model):
        self.agent = create_reconstruction_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class FittingAgent:
    """Histogram fitting and signal extraction agent."""

    def __init__(self, model):
        self.agent = create_fitting_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class StatisticalAnalysisAgent:
    """Statistical analysis and hypothesis testing agent."""

    def __init__(self, model):
        self.agent = create_statistical_analysis_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class PlottingAgent:
    """Plotting and visualization agent."""

    def __init__(self, model):
        self.agent = create_plotting_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class UncertaintyAgent:
    """Uncertainty quantification agent."""

    def __init__(self, model):
        self.agent = create_uncertainty_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)


class InterpretationAgent:
    """Results interpretation and paper writing agent."""

    def __init__(self, model):
        self.agent = create_interpretation_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)
