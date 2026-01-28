"""
Infrastructure Agents for HEP Analysis

Agents responsible for web research, data retrieval, and code search.
"""

from typing import Optional, List, Dict, Any
from smolagents import CodeAgent, ToolCallingAgent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.infrastructure import (
    WebSearchTool,
    TextParsingTool,
    InformationExtractionTool,
    ArxivSearchTool,
    InspireHEPSearchTool,
)
from tools.data_retrieval import (
    ZenodoDataTool,
    CERNOpenDataTool,
    ROOTFileLoaderTool,
    DatasetDownloaderTool,
)
from tools.code_tools import (
    GitHubSearchTool,
    GitLabSearchTool,
    CodeAnalysisTool,
    CodeGeneratorTool,
)


def create_web_research_agent(model, name: str = "web_research_agent"):
    """
    Create an agent for web research, paper search, and information extraction.

    This agent can:
    - Search arXiv and INSPIRE-HEP for physics papers
    - Extract measurements and parameters from papers
    - Parse physics information from text
    """
    tools = [
        WebSearchTool(),
        ArxivSearchTool(),
        InspireHEPSearchTool(),
        TextParsingTool(),
        InformationExtractionTool(),
    ]

    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Web research agent specialized in High Energy Physics literature.
        Can search arXiv, INSPIRE-HEP, and web for physics papers and information.
        Extracts measurements, cross sections, and analysis parameters from papers.
        Use this agent to find relevant papers, previous measurements, or theoretical predictions."""
    )

    return agent


def create_data_retrieval_agent(model, name: str = "data_retrieval_agent"):
    """
    Create an agent for downloading and loading physics data.

    This agent can:
    - Search and download data from Zenodo
    - Access CERN Open Data Portal
    - Load and inspect ROOT files
    - Download datasets from various sources
    """
    tools = [
        ZenodoDataTool(),
        CERNOpenDataTool(),
        ROOTFileLoaderTool(),
        DatasetDownloaderTool(),
    ]

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=20,
        name=name,
        description="""Data retrieval agent for HEP datasets.
        Can search and download data from Zenodo, CERN Open Data Portal.
        Loads and inspects ROOT files, extracts tree and histogram contents.
        Use this agent to obtain physics data for analysis.""",
        additional_authorized_imports=[
            "uproot", "numpy", "json", "os", "pathlib"
        ]
    )

    return agent


def create_code_search_agent(model, name: str = "code_search_agent"):
    """
    Create an agent for searching and analyzing physics analysis code.

    This agent can:
    - Search GitHub and GitLab for analysis code
    - Analyze Python code structure
    - Generate code templates
    """
    tools = [
        GitHubSearchTool(),
        GitLabSearchTool(),
        CodeAnalysisTool(),
        CodeGeneratorTool(),
    ]

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        name=name,
        description="""Code search and analysis agent for HEP.
        Searches GitHub and GitLab for physics analysis code and frameworks.
        Analyzes code structure, dependencies, and patterns.
        Generates code templates for common analysis tasks.
        Use this agent to find existing implementations or create new analysis code.""",
        additional_authorized_imports=[
            "ast", "json", "os", "pathlib", "re"
        ]
    )

    return agent


# Convenience class wrappers

class WebResearchAgent:
    """Web research agent for HEP literature and information."""

    def __init__(self, model):
        self.agent = create_web_research_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)

    def __call__(self, task: str, **kwargs) -> str:
        return self.run(task, **kwargs)


class DataRetrievalAgent:
    """Data retrieval agent for physics datasets."""

    def __init__(self, model):
        self.agent = create_data_retrieval_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)

    def __call__(self, task: str, **kwargs) -> str:
        return self.run(task, **kwargs)


class CodeSearchAgent:
    """Code search and generation agent."""

    def __init__(self, model):
        self.agent = create_code_search_agent(model)
        self.name = self.agent.name
        self.description = self.agent.description

    def run(self, task: str, **kwargs) -> str:
        return self.agent.run(task, **kwargs)

    def __call__(self, task: str, **kwargs) -> str:
        return self.run(task, **kwargs)
