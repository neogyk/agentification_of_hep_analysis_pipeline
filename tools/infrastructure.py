"""
Infrastructure Tools for HEP Analysis

Tools for web searching, text parsing, and information extraction
specific to particle physics research.
"""

import re
import json
from typing import Optional, List, Dict, Any
from smolagents import Tool


class WebSearchTool(Tool):
    """
    Web search tool for finding HEP-related information.
    Searches academic sources, preprint servers, and physics databases.
    """
    name = "hep_web_search"
    description = """
    Performs web searches focused on High Energy Physics content.
    Searches across arXiv, INSPIRE-HEP, CERN Document Server, and general web.
    Returns relevant results with titles, URLs, and snippets.

    Use this tool to find:
    - Published papers and preprints
    - Analysis techniques and methods
    - Experimental results and measurements
    - Theoretical predictions
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query (e.g., 'Higgs boson mass measurement CMS')"
        },
        "source": {
            "type": "string",
            "description": "Search source: 'all', 'arxiv', 'inspire', 'cds'. Default is 'all'"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 10)"
        }
    }
    output_type = "string"

    def forward(self, query: str, source: str = "all", max_results: int = 10) -> str:
        import requests
        from bs4 import BeautifulSoup

        results = []

        try:
            if source in ["all", "arxiv"]:
                arxiv_results = self._search_arxiv(query, max_results)
                results.extend(arxiv_results)
                print(f"Found {len(arxiv_results)} results from arXiv")

            if source in ["all", "inspire"]:
                inspire_results = self._search_inspire(query, max_results)
                results.extend(inspire_results)
                print(f"Found {len(inspire_results)} results from INSPIRE-HEP")

            if not results:
                return f"No results found for query: {query}"

            # Format results
            formatted = f"Search results for: {query}\n\n"
            for i, r in enumerate(results[:max_results], 1):
                formatted += f"{i}. {r['title']}\n"
                formatted += f"   Source: {r['source']}\n"
                formatted += f"   URL: {r['url']}\n"
                if r.get('authors'):
                    formatted += f"   Authors: {r['authors'][:200]}...\n" if len(r.get('authors', '')) > 200 else f"   Authors: {r['authors']}\n"
                if r.get('abstract'):
                    formatted += f"   Abstract: {r['abstract'][:300]}...\n" if len(r['abstract']) > 300 else f"   Abstract: {r['abstract']}\n"
                formatted += "\n"

            return formatted

        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _search_arxiv(self, query: str, max_results: int) -> List[Dict]:
        import requests
        import xml.etree.ElementTree as ET

        base_url = "http://export.arxiv.org/api/query"
        # Focus on hep-ex, hep-ph, hep-th categories
        params = {
            "search_query": f"all:{query} AND (cat:hep-ex OR cat:hep-ph OR cat:hep-th)",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)
                authors = entry.findall("atom:author/atom:name", ns)

                results.append({
                    "title": title.text.strip() if title is not None else "No title",
                    "abstract": summary.text.strip() if summary is not None else "",
                    "url": link.text if link is not None else "",
                    "authors": ", ".join([a.text for a in authors[:5]]),
                    "source": "arXiv"
                })

            return results
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []

    def _search_inspire(self, query: str, max_results: int) -> List[Dict]:
        import requests

        base_url = "https://inspirehep.net/api/literature"
        params = {
            "q": query,
            "size": max_results,
            "sort": "mostrecent"
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                metadata = hit.get("metadata", {})
                titles = metadata.get("titles", [{}])
                title = titles[0].get("title", "No title") if titles else "No title"

                abstracts = metadata.get("abstracts", [{}])
                abstract = abstracts[0].get("value", "") if abstracts else ""

                authors = metadata.get("authors", [])
                author_names = ", ".join([a.get("full_name", "") for a in authors[:5]])

                results.append({
                    "title": title,
                    "abstract": abstract,
                    "url": f"https://inspirehep.net/literature/{hit.get('id', '')}",
                    "authors": author_names,
                    "source": "INSPIRE-HEP"
                })

            return results
        except Exception as e:
            print(f"INSPIRE search error: {e}")
            return []


class TextParsingTool(Tool):
    """
    Parses and extracts structured information from physics papers and documents.
    """
    name = "text_parser"
    description = """
    Parses text from physics papers, notes, and documents.
    Extracts structured information like:
    - Measurement values with uncertainties
    - Cross sections and branching ratios
    - Kinematic cuts and selection criteria
    - Analysis parameters

    Input can be raw text, URL to a paper, or arXiv ID.
    """
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to parse, URL, or arXiv ID (e.g., '2312.12345')"
        },
        "extraction_type": {
            "type": "string",
            "description": "Type of info to extract: 'measurements', 'cuts', 'parameters', 'all'"
        }
    }
    output_type = "string"

    def forward(self, text: str, extraction_type: str = "all") -> str:
        import requests
        from bs4 import BeautifulSoup

        # Check if input is arXiv ID
        arxiv_pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
        if re.match(arxiv_pattern, text):
            text = self._fetch_arxiv_abstract(text)
            print(f"Fetched arXiv abstract for {text[:50]}...")

        # Check if input is URL
        if text.startswith("http"):
            text = self._fetch_url_content(text)
            print(f"Fetched content from URL")

        results = {"raw_text_length": len(text)}

        if extraction_type in ["all", "measurements"]:
            results["measurements"] = self._extract_measurements(text)

        if extraction_type in ["all", "cuts"]:
            results["cuts"] = self._extract_cuts(text)

        if extraction_type in ["all", "parameters"]:
            results["parameters"] = self._extract_parameters(text)

        return json.dumps(results, indent=2)

    def _fetch_arxiv_abstract(self, arxiv_id: str) -> str:
        import requests
        import xml.etree.ElementTree as ET

        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = requests.get(url, timeout=30)
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            summary = root.find(".//atom:summary", ns)
            return summary.text.strip() if summary is not None else ""
        except Exception as e:
            return f"Error fetching arXiv: {e}"

    def _fetch_url_content(self, url: str) -> str:
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            return f"Error fetching URL: {e}"

    def _extract_measurements(self, text: str) -> List[Dict]:
        """Extract physics measurements with uncertainties"""
        measurements = []

        # Pattern for values with uncertainties like "125.25 +/- 0.17 GeV"
        uncertainty_pattern = r"(\d+\.?\d*)\s*(?:\+/?-|±)\s*(\d+\.?\d*)\s*(?:\+/?-|±)?\s*(\d+\.?\d*)?\s*(GeV|MeV|TeV|pb|fb|nb)?"

        for match in re.finditer(uncertainty_pattern, text):
            measurements.append({
                "value": float(match.group(1)),
                "stat_uncertainty": float(match.group(2)),
                "syst_uncertainty": float(match.group(3)) if match.group(3) else None,
                "unit": match.group(4)
            })

        # Pattern for cross sections
        xs_pattern = r"(?:cross[- ]?section|σ)\s*[=:]\s*(\d+\.?\d*)\s*(?:\+/?-|±)\s*(\d+\.?\d*)\s*(pb|fb|nb)"
        for match in re.finditer(xs_pattern, text, re.IGNORECASE):
            measurements.append({
                "type": "cross_section",
                "value": float(match.group(1)),
                "uncertainty": float(match.group(2)),
                "unit": match.group(3)
            })

        return measurements

    def _extract_cuts(self, text: str) -> List[Dict]:
        """Extract kinematic cuts and selection criteria"""
        cuts = []

        # Pattern for pT cuts
        pt_pattern = r"p[_]?T\s*[><=]+\s*(\d+\.?\d*)\s*(GeV|MeV)?"
        for match in re.finditer(pt_pattern, text, re.IGNORECASE):
            cuts.append({
                "variable": "pT",
                "value": float(match.group(1)),
                "unit": match.group(2) or "GeV"
            })

        # Pattern for eta cuts
        eta_pattern = r"\|?[ηη]?\|?\s*[<>]=?\s*(\d+\.?\d*)"
        for match in re.finditer(eta_pattern, text):
            cuts.append({
                "variable": "eta",
                "value": float(match.group(1))
            })

        # Pattern for mass windows
        mass_pattern = r"(\d+\.?\d*)\s*<\s*[mM]_?(?:\w+)?\s*<\s*(\d+\.?\d*)\s*(GeV)?"
        for match in re.finditer(mass_pattern, text):
            cuts.append({
                "variable": "mass_window",
                "low": float(match.group(1)),
                "high": float(match.group(2)),
                "unit": match.group(3) or "GeV"
            })

        return cuts

    def _extract_parameters(self, text: str) -> List[Dict]:
        """Extract analysis parameters"""
        parameters = []

        # Luminosity
        lumi_pattern = r"(?:luminosity|L)\s*[=:]\s*(\d+\.?\d*)\s*(fb|pb|nb)[\^⁻]?[-]?1"
        for match in re.finditer(lumi_pattern, text, re.IGNORECASE):
            parameters.append({
                "name": "luminosity",
                "value": float(match.group(1)),
                "unit": f"{match.group(2)}^-1"
            })

        # Center of mass energy
        energy_pattern = r"(?:√s|sqrt\(s\)|center[- ]?of[- ]?mass)\s*[=:]\s*(\d+\.?\d*)\s*(TeV|GeV)"
        for match in re.finditer(energy_pattern, text, re.IGNORECASE):
            parameters.append({
                "name": "sqrt_s",
                "value": float(match.group(1)),
                "unit": match.group(2)
            })

        return parameters


class InformationExtractionTool(Tool):
    """
    Extracts specific physics information using pattern matching and NLP.
    """
    name = "info_extractor"
    description = """
    Extracts specific physics information from text using advanced pattern matching.
    Can extract:
    - Particle properties (mass, width, lifetime)
    - Decay channels and branching ratios
    - Production cross sections
    - Significance values and confidence levels
    - Systematic uncertainty sources
    """
    inputs = {
        "text": {
            "type": "string",
            "description": "Text to extract information from"
        },
        "info_type": {
            "type": "string",
            "description": "Type: 'particle_properties', 'decay_channels', 'cross_sections', 'significance', 'systematics'"
        }
    }
    output_type = "string"

    def forward(self, text: str, info_type: str = "all") -> str:
        extractors = {
            "particle_properties": self._extract_particle_properties,
            "decay_channels": self._extract_decay_channels,
            "cross_sections": self._extract_cross_sections,
            "significance": self._extract_significance,
            "systematics": self._extract_systematics,
        }

        if info_type == "all":
            results = {}
            for name, extractor in extractors.items():
                results[name] = extractor(text)
        else:
            results = {info_type: extractors.get(info_type, lambda x: [])(text)}

        return json.dumps(results, indent=2)

    def _extract_particle_properties(self, text: str) -> List[Dict]:
        """Extract particle mass, width, lifetime"""
        properties = []

        # Mass extraction
        mass_pattern = r"(?:mass|m_\w+)\s*[=:]\s*(\d+\.?\d*)\s*(?:\+/?-|±)\s*(\d+\.?\d*)\s*(GeV|MeV|TeV)"
        for match in re.finditer(mass_pattern, text, re.IGNORECASE):
            properties.append({
                "property": "mass",
                "value": float(match.group(1)),
                "uncertainty": float(match.group(2)),
                "unit": match.group(3)
            })

        # Width extraction
        width_pattern = r"(?:width|Γ|gamma)\s*[=:]\s*(\d+\.?\d*)\s*(?:\+/?-|±)?\s*(\d+\.?\d*)?\s*(GeV|MeV|keV)?"
        for match in re.finditer(width_pattern, text, re.IGNORECASE):
            properties.append({
                "property": "width",
                "value": float(match.group(1)),
                "uncertainty": float(match.group(2)) if match.group(2) else None,
                "unit": match.group(3) or "GeV"
            })

        return properties

    def _extract_decay_channels(self, text: str) -> List[Dict]:
        """Extract decay channels and branching ratios"""
        channels = []

        # Decay pattern like "H → γγ" or "H -> WW"
        decay_pattern = r"(\w+)\s*(?:→|->)\s*(\w+\s*\w*)"
        for match in re.finditer(decay_pattern, text):
            channels.append({
                "parent": match.group(1),
                "products": match.group(2).strip()
            })

        # Branching ratio
        br_pattern = r"(?:BR|branching\s*ratio)\s*\(\s*(\w+\s*→?\s*\w+)\s*\)\s*[=:]\s*(\d+\.?\d*(?:e[+-]?\d+)?)\s*(?:\+/?-|±)?\s*(\d+\.?\d*(?:e[+-]?\d+)?)?"
        for match in re.finditer(br_pattern, text, re.IGNORECASE):
            channels.append({
                "channel": match.group(1),
                "branching_ratio": float(match.group(2)),
                "uncertainty": float(match.group(3)) if match.group(3) else None
            })

        return channels

    def _extract_cross_sections(self, text: str) -> List[Dict]:
        """Extract production cross sections"""
        cross_sections = []

        xs_pattern = r"(?:σ|cross[- ]?section)\s*(?:\(\s*(\w+)\s*\))?\s*[=:]\s*(\d+\.?\d*)\s*(?:\+/?-|±)\s*(\d+\.?\d*)\s*(pb|fb|nb|mb)"
        for match in re.finditer(xs_pattern, text, re.IGNORECASE):
            cross_sections.append({
                "process": match.group(1) or "unknown",
                "value": float(match.group(2)),
                "uncertainty": float(match.group(3)),
                "unit": match.group(4)
            })

        return cross_sections

    def _extract_significance(self, text: str) -> List[Dict]:
        """Extract significance values"""
        significances = []

        # Sigma significance
        sigma_pattern = r"(\d+\.?\d*)\s*(?:σ|sigma|standard\s*deviations?)"
        for match in re.finditer(sigma_pattern, text, re.IGNORECASE):
            significances.append({
                "value": float(match.group(1)),
                "unit": "sigma"
            })

        # P-value
        pvalue_pattern = r"p[- ]?value\s*[=:]\s*(\d+\.?\d*(?:e[+-]?\d+)?)"
        for match in re.finditer(pvalue_pattern, text, re.IGNORECASE):
            significances.append({
                "p_value": float(match.group(1))
            })

        # Confidence level
        cl_pattern = r"(\d+\.?\d*)\s*%?\s*(?:C\.?L\.?|confidence\s*level)"
        for match in re.finditer(cl_pattern, text, re.IGNORECASE):
            significances.append({
                "confidence_level": float(match.group(1))
            })

        return significances

    def _extract_systematics(self, text: str) -> List[Dict]:
        """Extract systematic uncertainty sources"""
        systematics = []

        # Common systematic sources
        syst_keywords = [
            "jet energy scale", "JES", "jet energy resolution", "JER",
            "b-tagging", "trigger", "pileup", "luminosity",
            "PDF", "parton distribution", "theory", "scale variation",
            "lepton ID", "lepton isolation", "electron", "muon"
        ]

        for keyword in syst_keywords:
            pattern = rf"({keyword})\s*(?:uncertainty)?\s*[=:]?\s*(\d+\.?\d*)?\s*%?"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                systematics.append({
                    "source": match.group(1),
                    "value": float(match.group(2)) if match.group(2) else None,
                    "unit": "%"
                })

        return systematics


class ArxivSearchTool(Tool):
    """
    Dedicated arXiv search tool for physics preprints.
    """
    name = "arxiv_search"
    description = """
    Searches arXiv for physics preprints in hep-ex, hep-ph, hep-th categories.
    Returns paper titles, authors, abstracts, and arXiv IDs.
    Useful for finding recent analysis techniques and theoretical predictions.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query (supports arXiv search syntax)"
        },
        "category": {
            "type": "string",
            "description": "arXiv category: 'hep-ex', 'hep-ph', 'hep-th', 'all'. Default: 'all'"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results (default: 10, max: 50)"
        }
    }
    output_type = "string"

    def forward(self, query: str, category: str = "all", max_results: int = 10) -> str:
        import requests
        import xml.etree.ElementTree as ET

        max_results = min(max_results, 50)

        if category == "all":
            cat_filter = "(cat:hep-ex OR cat:hep-ph OR cat:hep-th)"
        else:
            cat_filter = f"cat:{category}"

        search_query = f"all:{query} AND {cat_filter}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        try:
            response = requests.get(
                "http://export.arxiv.org/api/query",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip()
                authors = [a.text for a in entry.findall("atom:author/atom:name", ns)]
                published = entry.find("atom:published", ns).text

                results.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors[:5],
                    "abstract": summary[:500],
                    "published": published[:10],
                    "url": f"https://arxiv.org/abs/{arxiv_id}"
                })

            if not results:
                return f"No arXiv results found for: {query}"

            formatted = f"arXiv search results for: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. [{r['arxiv_id']}] {r['title']}\n"
                formatted += f"   Authors: {', '.join(r['authors'])}\n"
                formatted += f"   Published: {r['published']}\n"
                formatted += f"   URL: {r['url']}\n"
                formatted += f"   Abstract: {r['abstract']}...\n\n"

            print(f"Found {len(results)} papers on arXiv")
            return formatted

        except Exception as e:
            return f"arXiv search error: {str(e)}"


class InspireHEPSearchTool(Tool):
    """
    Search INSPIRE-HEP database for published HEP papers.
    """
    name = "inspire_search"
    description = """
    Searches INSPIRE-HEP, the main database for High Energy Physics literature.
    Returns published papers with citation counts, DOIs, and journal references.
    Best for finding peer-reviewed publications and highly-cited papers.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query (supports INSPIRE search syntax)"
        },
        "sort": {
            "type": "string",
            "description": "Sort by: 'mostrecent', 'mostcited'. Default: 'mostrecent'"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results (default: 10)"
        }
    }
    output_type = "string"

    def forward(self, query: str, sort: str = "mostrecent", max_results: int = 10) -> str:
        import requests

        params = {
            "q": query,
            "size": max_results,
            "sort": sort,
            "fields": "titles,authors,abstracts,arxiv_eprints,dois,citation_count,publication_info"
        }

        try:
            response = requests.get(
                "https://inspirehep.net/api/literature",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                meta = hit.get("metadata", {})

                title = meta.get("titles", [{}])[0].get("title", "No title")
                authors = [a.get("full_name", "") for a in meta.get("authors", [])[:5]]
                abstract = meta.get("abstracts", [{}])[0].get("value", "")[:500]
                arxiv = meta.get("arxiv_eprints", [{}])[0].get("value", "")
                doi = meta.get("dois", [{}])[0].get("value", "") if meta.get("dois") else ""
                citations = meta.get("citation_count", 0)
                pub_info = meta.get("publication_info", [{}])[0] if meta.get("publication_info") else {}
                journal = pub_info.get("journal_title", "")
                year = pub_info.get("year", "")

                results.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "arxiv_id": arxiv,
                    "doi": doi,
                    "citations": citations,
                    "journal": journal,
                    "year": year,
                    "inspire_url": f"https://inspirehep.net/literature/{hit.get('id', '')}"
                })

            if not results:
                return f"No INSPIRE results found for: {query}"

            formatted = f"INSPIRE-HEP search results for: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. {r['title']}\n"
                formatted += f"   Authors: {', '.join(r['authors'])}\n"
                if r['journal']:
                    formatted += f"   Published: {r['journal']} ({r['year']})\n"
                if r['arxiv_id']:
                    formatted += f"   arXiv: {r['arxiv_id']}\n"
                if r['doi']:
                    formatted += f"   DOI: {r['doi']}\n"
                formatted += f"   Citations: {r['citations']}\n"
                formatted += f"   URL: {r['inspire_url']}\n\n"

            print(f"Found {len(results)} papers on INSPIRE-HEP")
            return formatted

        except Exception as e:
            return f"INSPIRE search error: {str(e)}"
