"""
Code Search and Analysis Tools for HEP Analysis

Tools for searching, analyzing, and generating physics analysis code.
"""

import os
import json
import re
from typing import Optional, List, Dict, Any
from pathlib import Path
from smolagents import Tool
import ast

class GitHubSearchTool(Tool):
    """
    Search GitHub for HEP analysis code and repositories.
    """
    name = "github_search"
    description = """
    Searches GitHub for physics analysis code, frameworks, and examples.
    Can search:
    - Repositories (e.g., CMSSW, athena, coffea)
    - Code snippets and files
    - Issues and discussions

    Useful for finding existing implementations, analysis templates,
    and community solutions.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "search_type": {
            "type": "string",
            "description": "Type: 'repositories', 'code', 'issues'. Default: 'repositories'"
        },
        "language": {
            "type": "string",
            "description": "Filter by language: 'python', 'cpp', 'all'. Default: 'all'"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results (default: 10)"
        }
    }
    output_type = "string"

    def forward(self, query: str, search_type: str = "repositories",
                language: str = "all", max_results: int = 10) -> str:
        import requests

        headers = {}
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        if search_type == "repositories":
            return self._search_repos(query, language, max_results, headers)
        elif search_type == "code":
            return self._search_code(query, language, max_results, headers)
        elif search_type == "issues":
            return self._search_issues(query, max_results, headers)
        else:
            return f"Unknown search type: {search_type}"

    def _search_repos(self, query: str, language: str, max_results: int, headers: dict) -> str:
        import requests

        search_query = f"{query} topic:physics OR topic:hep OR topic:particle-physics"
        if language != "all":
            search_query += f" language:{language}"

        try:
            response = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": search_query, "per_page": max_results, "sort": "stars"},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for repo in data.get("items", []):
                results.append({
                    "name": repo["full_name"],
                    "description": repo.get("description", "")[:200],
                    "stars": repo["stargazers_count"],
                    "language": repo.get("language"),
                    "url": repo["html_url"],
                    "topics": repo.get("topics", [])[:5],
                    "updated": repo["updated_at"][:10]
                })

            if not results:
                return f"No GitHub repositories found for: {query}"

            formatted = f"GitHub repository search: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. {r['name']} ⭐ {r['stars']}\n"
                formatted += f"   {r['description']}\n"
                formatted += f"   Language: {r['language']}, Updated: {r['updated']}\n"
                formatted += f"   Topics: {', '.join(r['topics'])}\n"
                formatted += f"   URL: {r['url']}\n\n"

            print(f"Found {len(results)} repositories")
            return formatted

        except Exception as e:
            return f"GitHub search error: {str(e)}"

    def _search_code(self, query: str, language: str, max_results: int, headers: dict) -> str:
        import requests

        search_query = query
        if language != "all":
            search_query += f" language:{language}"

        try:
            response = requests.get(
                "https://api.github.com/search/code",
                params={"q": search_query, "per_page": max_results},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "name": item["name"],
                    "path": item["path"],
                    "repository": item["repository"]["full_name"],
                    "url": item["html_url"],
                    "score": item.get("score", 0)
                })

            if not results:
                return f"No code found for: {query}"

            formatted = f"GitHub code search: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. {r['name']}\n"
                formatted += f"   Repository: {r['repository']}\n"
                formatted += f"   Path: {r['path']}\n"
                formatted += f"   URL: {r['url']}\n\n"

            print(f"Found {len(results)} code files")
            return formatted

        except Exception as e:
            return f"GitHub code search error: {str(e)}"

    def _search_issues(self, query: str, max_results: int, headers: dict) -> str:
        import requests

        try:
            response = requests.get(
                "https://api.github.com/search/issues",
                params={"q": query, "per_page": max_results, "sort": "updated"},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for issue in data.get("items", []):
                results.append({
                    "title": issue["title"],
                    "state": issue["state"],
                    "repository": issue["repository_url"].split("/")[-2] + "/" + issue["repository_url"].split("/")[-1],
                    "url": issue["html_url"],
                    "created": issue["created_at"][:10],
                    "comments": issue["comments"]
                })

            if not results:
                return f"No issues found for: {query}"

            formatted = f"GitHub issues search: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. [{r['state']}] {r['title']}\n"
                formatted += f"   Repository: {r['repository']}\n"
                formatted += f"   Created: {r['created']}, Comments: {r['comments']}\n"
                formatted += f"   URL: {r['url']}\n\n"

            return formatted

        except Exception as e:
            return f"GitHub issues search error: {str(e)}"


class GitLabSearchTool(Tool):
    """
    Search GitLab for HEP analysis code (including CERN GitLab).
    """
    name = "gitlab_search"
    description = """
    Searches GitLab instances for physics analysis code.
    Includes CERN GitLab (gitlab.cern.ch) for official experiment code.

    Useful for finding:
    - Official analysis frameworks
    - Experiment-specific code
    - Internal tools and utilities
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "instance": {
            "type": "string",
            "description": "GitLab instance: 'cern', 'gitlab.com'. Default: 'gitlab.com'"
        },
        "search_type": {
            "type": "string",
            "description": "Type: 'projects', 'blobs' (code). Default: 'projects'"
        }
    }
    output_type = "string"

    def forward(self, query: str, instance: str = "gitlab.com",
                search_type: str = "projects") -> str:
        import requests

        if instance == "cern":
            base_url = "https://gitlab.cern.ch/api/v4"
        else:
            base_url = "https://gitlab.com/api/v4"

        gitlab_token = os.getenv("GITLAB_TOKEN")
        headers = {}
        if gitlab_token:
            headers["PRIVATE-TOKEN"] = gitlab_token

        try:
            if search_type == "projects":
                response = requests.get(
                    f"{base_url}/projects",
                    params={"search": query, "per_page": 10},
                    headers=headers,
                    timeout=30
                )
            else:  # blobs
                response = requests.get(
                    f"{base_url}/search",
                    params={"scope": "blobs", "search": query, "per_page": 10},
                    headers=headers,
                    timeout=30
                )

            response.raise_for_status()
            data = response.json()

            if search_type == "projects":
                results = []
                for proj in data:
                    results.append({
                        "name": proj.get("path_with_namespace"),
                        "description": (proj.get("description") or "")[:200],
                        "stars": proj.get("star_count", 0),
                        "url": proj.get("web_url"),
                        "visibility": proj.get("visibility"),
                        "updated": proj.get("last_activity_at", "")[:10]
                    })

                if not results:
                    return f"No GitLab projects found for: {query}"

                formatted = f"GitLab ({instance}) project search: {query}\n\n"
                for i, r in enumerate(results, 1):
                    formatted += f"{i}. {r['name']} ⭐ {r['stars']}\n"
                    formatted += f"   {r['description']}\n"
                    formatted += f"   Visibility: {r['visibility']}, Updated: {r['updated']}\n"
                    formatted += f"   URL: {r['url']}\n\n"

                return formatted

            else:  # Code search results
                results = []
                for blob in data:
                    results.append({
                        "filename": blob.get("filename"),
                        "path": blob.get("path"),
                        "project_id": blob.get("project_id"),
                        "data": blob.get("data", "")[:200]
                    })

                formatted = f"GitLab ({instance}) code search: {query}\n\n"
                for i, r in enumerate(results, 1):
                    formatted += f"{i}. {r['filename']}\n"
                    formatted += f"   Path: {r['path']}\n"
                    formatted += f"   Preview: {r['data'][:100]}...\n\n"

                return formatted

        except Exception as e:
            return f"GitLab search error: {str(e)}"


class CodeAnalysisTool(Tool):
    """
    Analyzes physics analysis code for patterns and structure.
    """
    name = "code_analyzer"
    description = """
    Analyzes physics analysis code to extract:
    - Function signatures and documentation
    - Class hierarchies
    - Import dependencies
    - Analysis workflow structure
    - Variable definitions and data flow

    Useful for understanding existing code before modification or integration.
    """
    inputs = {
        "code": {
            "type": "string",
            "description": "Code to analyze (or file path)"
        },
        "analysis_type": {
            "type": "string",
            "description": "Type: 'structure', 'dependencies', 'functions', 'classes', 'all'"
        }
    }
    output_type = "string"

    def forward(self, code: str, analysis_type: str = "all") -> str:
        import ast

        # Check if code is a file path
        if os.path.exists(code):
            with open(code, 'r') as f:
                code = f.read()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error in code: {str(e)}"

        results = {}

        if analysis_type in ["all", "structure"]:
            results["structure"] = self._analyze_structure(tree)

        if analysis_type in ["all", "dependencies"]:
            results["dependencies"] = self._analyze_dependencies(tree)

        if analysis_type in ["all", "functions"]:
            results["functions"] = self._analyze_functions(tree)

        if analysis_type in ["all", "classes"]:
            results["classes"] = self._analyze_classes(tree)

        return json.dumps(results, indent=2)

    def _analyze_structure(self, tree: ast.AST) -> Dict:
        """Analyze overall code structure"""
        structure = {
            "n_lines": len(ast.unparse(tree).split('\n')),
            "n_functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            "n_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            "n_imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
            "top_level_items": []
        }

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                structure["top_level_items"].append(f"function: {node.name}")
            elif isinstance(node, ast.ClassDef):
                structure["top_level_items"].append(f"class: {node.name}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        structure["top_level_items"].append(f"variable: {target.id}")

        return structure

    def _analyze_dependencies(self, tree: ast.AST) -> Dict:
        """Analyze import dependencies"""
        imports = {"standard_lib": [], "third_party": [], "local": []}

        standard_libs = {
            'os', 'sys', 'json', 'math', 'time', 'datetime', 're', 'collections',
            'itertools', 'functools', 'pathlib', 'typing', 'logging', 'argparse'
        }

        hep_libs = {
            'uproot', 'awkward', 'vector', 'hist', 'coffea', 'pyhf', 'iminuit',
            'particle', 'hepunits', 'mplhep', 'ROOT'
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in standard_libs:
                        imports["standard_lib"].append(alias.name)
                    elif module in hep_libs:
                        imports["third_party"].append({"name": alias.name, "type": "hep"})
                    else:
                        imports["third_party"].append({"name": alias.name, "type": "general"})

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    imported = [alias.name for alias in node.names]
                    if module in standard_libs:
                        imports["standard_lib"].append(f"{node.module}: {imported}")
                    elif module in hep_libs:
                        imports["third_party"].append({
                            "name": node.module,
                            "items": imported,
                            "type": "hep"
                        })
                    elif module.startswith('.'):
                        imports["local"].append(f"{node.module}: {imported}")
                    else:
                        imports["third_party"].append({
                            "name": node.module,
                            "items": imported,
                            "type": "general"
                        })

        return imports

    def _analyze_functions(self, tree: ast.AST) -> List[Dict]:
        """Analyze function definitions"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "defaults": len(node.args.defaults),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node),
                    "returns": ast.unparse(node.returns) if node.returns else None,
                    "line": node.lineno
                }

                # Check for physics-related patterns
                func_info["physics_patterns"] = self._detect_physics_patterns(node)

                functions.append(func_info)

        return functions

    def _analyze_classes(self, tree: ast.AST) -> List[Dict]:
        """Analyze class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [ast.unparse(b) for b in node.bases],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "attributes": [],
                    "line": node.lineno
                }

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info["methods"].append(item.name)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_info["attributes"].append(target.id)

                classes.append(class_info)

        return classes

    def _detect_physics_patterns(self, node: ast.FunctionDef) -> List[str]:
        """Detect common physics analysis patterns"""
        patterns = []
        code = ast.unparse(node)

        physics_patterns = {
            "event_selection": ["cut", "selection", "filter", "mask", "pt >", "eta <"],
            "histogram_filling": ["hist", "fill", "histogram", "bin"],
            "fitting": ["fit", "minimize", "likelihood", "chi2"],
            "systematic": ["systematic", "uncertainty", "variation", "up", "down"],
            "plotting": ["plot", "draw", "canvas", "figure", "ax."],
            "data_loading": ["uproot", "open", "read", "load"]
        }

        for pattern_name, keywords in physics_patterns.items():
            for keyword in keywords:
                if keyword.lower() in code.lower():
                    patterns.append(pattern_name)
                    break

        return list(set(patterns))


class CodeGeneratorTool(Tool):
    """
    Generates physics analysis code based on specifications.
    """
    name = "code_generator"
    description = """
    Generates physics analysis code templates and implementations.
    Can generate:
    - Event selection code
    - Histogram filling and fitting
    - Systematic uncertainty calculations
    - Plotting code (mplhep style)
    - Analysis configuration files
    """
    inputs = {
        "code_type": {
            "type": "string",
            "description": "Type: 'event_selection', 'histogram', 'fit', 'systematic', 'plot', 'config'"
        },
        "specifications": {
            "type": "string",
            "description": "JSON string with specifications for the code"
        }
    }
    output_type = "string"

    def forward(self, code_type: str, specifications: str) -> str:
        try:
            specs = json.loads(specifications)
        except json.JSONDecodeError:
            return f"Invalid JSON specifications: {specifications}"

        generators = {
            "event_selection": self._generate_event_selection,
            "histogram": self._generate_histogram,
            "fit": self._generate_fit,
            "systematic": self._generate_systematic,
            "plot": self._generate_plot,
            "config": self._generate_config
        }

        if code_type not in generators:
            return f"Unknown code type: {code_type}"

        return generators[code_type](specs)

    def _generate_event_selection(self, specs: Dict) -> str:
        """Generate event selection code"""
        cuts = specs.get("cuts", [])
        variables = specs.get("variables", [])

        code = '''"""
Event Selection Module
Generated for HEP analysis
"""
import numpy as np
import awkward as ak


def apply_event_selection(events):
    """
    Apply event selection cuts.

    Args:
        events: Awkward array with event data

    Returns:
        Selected events passing all cuts
    """
    # Initialize mask
    mask = ak.ones_like(events.event, dtype=bool)

'''
        for cut in cuts:
            var = cut.get("variable", "pt")
            op = cut.get("operator", ">")
            value = cut.get("value", 0)
            code += f'    # {cut.get("description", f"{var} cut")}\n'
            code += f'    mask = mask & (events.{var} {op} {value})\n\n'

        code += '''    # Apply mask
    selected = events[mask]

    print(f"Selection efficiency: {ak.sum(mask) / len(mask) * 100:.1f}%")
    print(f"Events passing: {ak.sum(mask)} / {len(mask)}")

    return selected


def get_cut_flow(events):
    """Generate cut flow table"""
    cut_flow = {"Initial": len(events)}

'''
        for i, cut in enumerate(cuts):
            var = cut.get("variable", "pt")
            op = cut.get("operator", ">")
            value = cut.get("value", 0)
            name = cut.get("name", f"Cut_{i}")
            code += f'    mask_{i} = events.{var} {op} {value}\n'
            code += f'    cut_flow["{name}"] = ak.sum(mask_{i})\n'

        code += '''
    return cut_flow
'''
        return code

    def _generate_histogram(self, specs: Dict) -> str:
        """Generate histogram code"""
        variable = specs.get("variable", "mass")
        bins = specs.get("bins", 50)
        range_low = specs.get("range", [0, 100])[0]
        range_high = specs.get("range", [0, 100])[1]
        label = specs.get("label", variable)

        code = f'''"""
Histogram Module
Generated for HEP analysis
"""
import numpy as np
import hist
from hist import Hist


def create_histogram():
    """Create histogram for {variable}"""
    h = Hist(
        hist.axis.Regular({bins}, {range_low}, {range_high}, name="{variable}", label="{label}"),
        storage=hist.storage.Weight()
    )
    return h


def fill_histogram(h, data, weights=None):
    """Fill histogram with data"""
    if weights is None:
        weights = np.ones(len(data))

    h.fill({variable}=data, weight=weights)

    print(f"Filled {{len(data)}} events")
    print(f"Integral: {{h.sum():.1f}}")

    return h


def get_histogram_stats(h):
    """Calculate histogram statistics"""
    centers = h.axes[0].centers
    values = h.values()
    variances = h.variances()

    total = np.sum(values)
    mean = np.average(centers, weights=values)
    variance = np.average((centers - mean)**2, weights=values)

    return {{
        "entries": total,
        "mean": mean,
        "std": np.sqrt(variance),
        "underflow": h.values(flow=True)[0],
        "overflow": h.values(flow=True)[-1]
    }}
'''
        return code

    def _generate_fit(self, specs: Dict) -> str:
        """Generate fitting code"""
        model = specs.get("model", "gaussian")
        variable = specs.get("variable", "mass")

        code = f'''"""
Fitting Module
Generated for HEP analysis
"""
import numpy as np
from scipy import stats
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, BinnedNLL


def {model}_pdf(x, *params):
    """
    {model.capitalize()} probability density function
    """
'''
        if model == "gaussian":
            code += '''    mu, sigma = params
    return stats.norm.pdf(x, mu, sigma)
'''
        elif model == "breit_wigner":
            code += '''    mass, width = params
    return stats.cauchy.pdf(x, mass, width/2)
'''
        elif model == "crystal_ball":
            code += '''    mu, sigma, alpha, n = params
    # Crystal Ball function implementation
    t = (x - mu) / sigma
    abs_alpha = np.abs(alpha)

    a = (n / abs_alpha) ** n * np.exp(-0.5 * abs_alpha**2)
    b = n / abs_alpha - abs_alpha

    result = np.where(
        t > -alpha,
        np.exp(-0.5 * t**2),
        a * (b - t) ** (-n)
    )
    return result / (sigma * np.sqrt(2 * np.pi))
'''
        else:  # exponential background
            code += '''    slope = params[0]
    return np.exp(slope * x)
'''

        code += f'''

def fit_histogram(histogram, initial_params):
    """
    Fit histogram with {model} model

    Args:
        histogram: hist.Hist object
        initial_params: dict with initial parameter values

    Returns:
        Minuit fit result
    """
    def model(x, *params):
        return {model}_pdf(x, *params)

    # Create cost function
    cost = BinnedNLL(histogram.values(), histogram.axes[0].edges, model)

    # Initialize Minuit
    m = Minuit(cost, **initial_params)

    # Set parameter limits if provided
    # m.limits["mu"] = (80, 100)

    # Run fit
    m.migrad()
    m.hesse()

    print("Fit Results:")
    print(f"  Valid: {{m.valid}}")
    print(f"  FCN: {{m.fval:.2f}}")
    for p in m.parameters:
        print(f"  {{p}}: {{m.values[p]:.4f}} +/- {{m.errors[p]:.4f}}")

    return m


def calculate_significance(signal_hist, background_hist, signal_region):
    """
    Calculate signal significance

    Args:
        signal_hist: Signal histogram
        background_hist: Background histogram
        signal_region: Tuple (low, high) for signal window

    Returns:
        Significance in sigma
    """
    low, high = signal_region

    # Integrate in signal region
    s = signal_hist[hist.loc(low):hist.loc(high)].sum().value
    b = background_hist[hist.loc(low):hist.loc(high)].sum().value

    # Simple S/sqrt(B) significance
    if b > 0:
        significance = s / np.sqrt(b)
    else:
        significance = 0

    print(f"Signal: {{s:.1f}}, Background: {{b:.1f}}")
    print(f"Significance: {{significance:.2f}} sigma")

    return significance
'''
        return code

    def _generate_systematic(self, specs: Dict) -> str:
        """Generate systematic uncertainty code"""
        sources = specs.get("sources", ["jet_energy_scale", "luminosity"])

        code = '''"""
Systematic Uncertainty Module
Generated for HEP analysis
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SystematicVariation:
    """Container for systematic variation"""
    name: str
    up: float
    down: float
    symmetric: bool = True

    @property
    def value(self) -> float:
        """Get symmetric uncertainty"""
        return (abs(self.up) + abs(self.down)) / 2


class SystematicCalculator:
    """Calculate systematic uncertainties"""

    def __init__(self):
        self.variations: Dict[str, SystematicVariation] = {}

'''
        for source in sources:
            code += f'''
    def calculate_{source}(self, nominal, varied_up, varied_down=None):
        """Calculate {source} systematic"""
        up = (varied_up - nominal) / nominal if nominal != 0 else 0
        down = (varied_down - nominal) / nominal if varied_down and nominal != 0 else -up

        self.variations["{source}"] = SystematicVariation(
            name="{source}",
            up=up,
            down=down
        )
        return self.variations["{source}"]
'''

        code += '''

    def get_total_uncertainty(self, correlation_matrix=None):
        """
        Calculate total systematic uncertainty

        Args:
            correlation_matrix: Optional correlation matrix between sources

        Returns:
            Total up and down uncertainties
        """
        if correlation_matrix is None:
            # Assume uncorrelated - add in quadrature
            total_up = np.sqrt(sum(v.up**2 for v in self.variations.values()))
            total_down = np.sqrt(sum(v.down**2 for v in self.variations.values()))
        else:
            # Use correlation matrix
            names = list(self.variations.keys())
            ups = np.array([self.variations[n].up for n in names])
            downs = np.array([self.variations[n].down for n in names])

            total_up = np.sqrt(ups @ correlation_matrix @ ups)
            total_down = np.sqrt(downs @ correlation_matrix @ downs)

        return total_up, total_down

    def print_summary(self):
        """Print systematic uncertainty summary"""
        print("Systematic Uncertainties:")
        print("-" * 40)
        for name, var in self.variations.items():
            print(f"  {name:25s}: +{var.up*100:5.2f}% / {var.down*100:5.2f}%")

        total_up, total_down = self.get_total_uncertainty()
        print("-" * 40)
        print(f"  {'Total':25s}: +{total_up*100:5.2f}% / {total_down*100:5.2f}%")
'''
        return code

    def _generate_plot(self, specs: Dict) -> str:
        """Generate plotting code"""
        plot_type = specs.get("type", "histogram")
        style = specs.get("style", "CMS")

        code = f'''"""
Plotting Module
Generated for HEP analysis - {style} style
"""
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Set style
hep.style.use("{style}")


def plot_histogram(histogram, label="Data", color="black", **kwargs):
    """
    Plot histogram with {style} style

    Args:
        histogram: hist.Hist object
        label: Legend label
        color: Plot color
        **kwargs: Additional plotting arguments
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    hep.histplot(
        histogram,
        ax=ax,
        label=label,
        color=color,
        histtype="errorbar",
        **kwargs
    )

    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel("Events")
    ax.legend()

    # Add {style} label
    hep.{style.lower()}.label(ax=ax, data=True, lumi=139)

    return fig, ax


def plot_data_mc_comparison(data_hist, mc_hists, mc_labels, mc_colors):
    """
    Plot data/MC comparison with ratio panel

    Args:
        data_hist: Data histogram
        mc_hists: List of MC histograms
        mc_labels: Labels for MC samples
        mc_colors: Colors for MC samples
    """
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={{"height_ratios": [3, 1]}},
        sharex=True
    )

    # Stack MC histograms
    hep.histplot(
        mc_hists,
        ax=ax,
        label=mc_labels,
        color=mc_colors,
        stack=True,
        histtype="fill"
    )

    # Plot data points
    hep.histplot(
        data_hist,
        ax=ax,
        label="Data",
        color="black",
        histtype="errorbar"
    )

    ax.set_ylabel("Events")
    ax.legend()
    hep.{style.lower()}.label(ax=ax, data=True, lumi=139)

    # Ratio panel
    mc_total = sum(h.values() for h in mc_hists)
    ratio = data_hist.values() / mc_total
    ratio_err = np.sqrt(data_hist.variances()) / mc_total

    centers = data_hist.axes[0].centers
    rax.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", markersize=4)
    rax.axhline(1, color="gray", linestyle="--")
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(data_hist.axes[0].label)
    rax.set_ylim(0.5, 1.5)

    plt.tight_layout()
    return fig, (ax, rax)


def plot_fit_result(histogram, fit_result, model_func):
    """
    Plot histogram with fit result overlay

    Args:
        histogram: Data histogram
        fit_result: Minuit fit result
        model_func: Model function
    """
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={{"height_ratios": [3, 1]}},
        sharex=True
    )

    # Data
    hep.histplot(histogram, ax=ax, label="Data", histtype="errorbar")

    # Fit curve
    x = np.linspace(histogram.axes[0].edges[0], histogram.axes[0].edges[-1], 200)
    y = model_func(x, *fit_result.values)
    # Scale to histogram
    bin_width = histogram.axes[0].edges[1] - histogram.axes[0].edges[0]
    y_scaled = y * histogram.sum().value * bin_width

    ax.plot(x, y_scaled, "r-", label="Fit", linewidth=2)

    ax.set_ylabel("Events")
    ax.legend()

    # Residuals
    expected = model_func(histogram.axes[0].centers, *fit_result.values) * histogram.sum().value * bin_width
    residuals = (histogram.values() - expected) / np.sqrt(histogram.variances())

    rax.errorbar(histogram.axes[0].centers, residuals, yerr=1, fmt="ko", markersize=4)
    rax.axhline(0, color="gray", linestyle="--")
    rax.set_ylabel("Pull")
    rax.set_xlabel(histogram.axes[0].label)
    rax.set_ylim(-4, 4)

    plt.tight_layout()
    return fig, (ax, rax)


def save_plot(fig, filename, formats=["pdf", "png"]):
    """Save plot in multiple formats"""
    for fmt in formats:
        fig.savefig(f"{{filename}}.{{fmt}}", dpi=300, bbox_inches="tight")
        print(f"Saved: {{filename}}.{{fmt}}")
'''
        return code

    def _generate_config(self, specs: Dict) -> str:
        """Generate analysis configuration"""
        analysis_name = specs.get("name", "analysis")
        channel = specs.get("channel", "mu")

        config = f'''# Analysis Configuration
# Generated for {analysis_name}

analysis:
  name: "{analysis_name}"
  channel: "{channel}"
  version: "1.0.0"

data:
  luminosity: 139.0  # fb^-1
  sqrt_s: 13  # TeV
  datasets:
    - name: "data"
      path: "data/data.root"
      tree: "Events"

mc_samples:
  - name: "signal"
    path: "mc/signal.root"
    cross_section: 1.0  # pb
    k_factor: 1.0

  - name: "background_1"
    path: "mc/bkg1.root"
    cross_section: 100.0
    k_factor: 1.1

selection:
  preselection:
    - variable: "n_leptons"
      operator: ">="
      value: 1

  signal_region:
    - variable: "lepton_pt"
      operator: ">"
      value: 25.0
      unit: "GeV"

    - variable: "abs(lepton_eta)"
      operator: "<"
      value: 2.5

    - variable: "met"
      operator: ">"
      value: 20.0
      unit: "GeV"

histograms:
  - name: "mass"
    variable: "invariant_mass"
    bins: 60
    range: [60, 120]
    label: "$m_{{\\ell\\ell}}$ [GeV]"

  - name: "pt"
    variable: "lepton_pt"
    bins: 50
    range: [0, 200]
    label: "$p_{{T}}$ [GeV]"

systematics:
  experimental:
    - name: "jet_energy_scale"
      type: "shape"
      variations: ["up", "down"]

    - name: "luminosity"
      type: "normalization"
      value: 0.017

  theoretical:
    - name: "pdf"
      type: "shape"
      n_variations: 100

    - name: "scale"
      type: "envelope"
      variations: ["muR_up", "muR_down", "muF_up", "muF_down"]

fit:
  model: "signal_plus_background"
  signal_model: "gaussian"
  background_model: "exponential"
  fit_range: [70, 110]
  poi: "mu"  # parameter of interest

output:
  directory: "results/{analysis_name}"
  formats: ["pdf", "png", "root"]
'''
        return config
