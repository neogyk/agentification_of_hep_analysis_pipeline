"""
Configuration settings for HEP Analysis Multi-Agent System
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """LLM Model configuration"""
    provider: str = "anthropic"  # anthropic, openai, huggingface
    model_id: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class DataConfig:
    """Data storage and retrieval configuration"""
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    zenodo_token: Optional[str] = field(default_factory=lambda: os.getenv("ZENODO_TOKEN"))
    cern_open_data_url: str = "http://opendata.cern.ch/api/records"


@dataclass
class AnalysisConfig:
    """Physics analysis configuration"""
    # Event selection
    default_pt_cut: float = 25.0  # GeV
    default_eta_cut: float = 2.5
    default_isolation_cut: float = 0.1

    # Fitting
    fit_range: tuple = (60.0, 120.0)  # GeV for Z mass
    n_bins: int = 60

    # Statistical analysis
    confidence_level: float = 0.95
    n_toys: int = 1000

    # Systematics
    systematic_sources: List[str] = field(default_factory=lambda: [
        "jet_energy_scale",
        "jet_energy_resolution",
        "muon_scale",
        "electron_scale",
        "pileup",
        "luminosity",
        "pdf",
        "theory_scale"
    ])


@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    max_steps: int = 20
    planning_interval: int = 3
    verbose: bool = True
    log_level: str = "INFO"
    authorized_imports: List[str] = field(default_factory=lambda: [
        "numpy", "pandas", "scipy", "matplotlib", "uproot",
        "awkward", "hist", "iminuit", "pyhf", "mplhep",
        "vector", "particle", "json", "os", "pathlib",
        "collections", "itertools", "functools", "time",
        "datetime", "re", "math", "statistics"
    ])


@dataclass
class HEPAnalysisConfig:
    """Main configuration container"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self):
        """Create necessary directories"""
        for dir_path in [self.data.data_dir, self.data.cache_dir, self.data.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Particle physics constants
PARTICLE_MASSES = {
    "electron": 0.000511,  # GeV
    "muon": 0.10566,       # GeV
    "tau": 1.777,          # GeV
    "Z": 91.1876,          # GeV
    "W": 80.379,           # GeV
    "H": 125.25,           # GeV
    "top": 172.76,         # GeV
}

# Standard Model cross sections (pb) at 13 TeV
SM_CROSS_SECTIONS = {
    "Z_to_ll": 1921.0,
    "W_to_lnu": 20508.9,
    "ttbar": 831.76,
    "WW": 118.7,
    "WZ": 47.13,
    "ZZ": 16.91,
    "H_ggF": 48.58,
    "H_VBF": 3.782,
}

# PDG particle IDs
PDG_IDS = {
    "electron": 11,
    "muon": 13,
    "tau": 15,
    "photon": 22,
    "Z": 23,
    "W+": 24,
    "W-": -24,
    "H": 25,
    "gluon": 21,
}
