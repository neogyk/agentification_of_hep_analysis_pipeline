"""
HEP Analysis Tools Package

This package contains all tools used by the HEP analysis multi-agent system.
"""

from .infrastructure import (
    WebSearchTool,
    TextParsingTool,
    InformationExtractionTool,
    ArxivSearchTool,
    InspireHEPSearchTool,
)

from .data_retrieval import (
    ZenodoDataTool,
    CERNOpenDataTool,
    ROOTFileLoaderTool,
    DatasetDownloaderTool,
)

# Dedicated data fetchers with full functionality
from .data_fetchers import (
    CERNOpenDataFetcher,
    ZenodoFetcher,
    FileInfo,
    DatasetInfo,
    DownloadResult,
    search_cern_opendata,
    search_zenodo,
    download_from_cern,
    download_from_zenodo,
)

from .code_tools import (
    GitHubSearchTool,
    GitLabSearchTool,
    CodeAnalysisTool,
    CodeGeneratorTool,
)

from .physics_tools import (
    EventSelectionTool,
    DataProcessingTool,
    ReconstructionTool,
    KinematicCalculatorTool,
)

from .fitting_tools import (
    HistogramFitterTool,
    BackgroundModelTool,
    SignalModelTool,
    StatisticalFitTool,
)

from .plotting_tools import (
    HistogramPlotTool,
    FitResultPlotTool,
    ComparisonPlotTool,
    PublicationPlotTool,
)

from .uncertainty_tools import (
    SystematicUncertaintyTool,
    StatisticalUncertaintyTool,
    UncertaintyCombinerTool,
    CovarianceMatrixTool,
)

from .interpretation_tools import (
    ResultsInterpreterTool,
    SignificanceCalculatorTool,
    LimitCalculatorTool,
    PaperWriterTool,
)

__all__ = [
    # Infrastructure
    "WebSearchTool",
    "TextParsingTool",
    "InformationExtractionTool",
    "ArxivSearchTool",
    "InspireHEPSearchTool",
    # Data retrieval (smolagents Tools)
    "ZenodoDataTool",
    "CERNOpenDataTool",
    "ROOTFileLoaderTool",
    "DatasetDownloaderTool",
    # Data fetchers (standalone classes)
    "CERNOpenDataFetcher",
    "ZenodoFetcher",
    "FileInfo",
    "DatasetInfo",
    "DownloadResult",
    "search_cern_opendata",
    "search_zenodo",
    "download_from_cern",
    "download_from_zenodo",
    # Code tools
    "GitHubSearchTool",
    "GitLabSearchTool",
    "CodeAnalysisTool",
    "CodeGeneratorTool",
    # Physics tools
    "EventSelectionTool",
    "DataProcessingTool",
    "ReconstructionTool",
    "KinematicCalculatorTool",
    # Fitting tools
    "HistogramFitterTool",
    "BackgroundModelTool",
    "SignalModelTool",
    "StatisticalFitTool",
    # Plotting tools
    "HistogramPlotTool",
    "FitResultPlotTool",
    "ComparisonPlotTool",
    "PublicationPlotTool",
    # Uncertainty tools
    "SystematicUncertaintyTool",
    "StatisticalUncertaintyTool",
    "UncertaintyCombinerTool",
    "CovarianceMatrixTool",
    # Interpretation tools
    "ResultsInterpreterTool",
    "SignificanceCalculatorTool",
    "LimitCalculatorTool",
    "PaperWriterTool",
]
