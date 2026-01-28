"""
Data Retrieval Tools for HEP Analysis

Tools for downloading and loading physics data from various sources:
- Zenodo
- CERN Open Data Portal
- ROOT files
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from smolagents import Tool


class ZenodoDataTool(Tool):
    """
    Tool for searching and downloading data from Zenodo.
    """
    name = "zenodo_data"
    description = """
    Searches and downloads datasets from Zenodo repository.
    Zenodo hosts physics datasets, analysis code, and supplementary materials.

    Can:
    - Search for datasets by keywords
    - Get dataset metadata and file lists
    - Download specific files or entire datasets
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Action: 'search', 'info', 'download'"
        },
        "query": {
            "type": "string",
            "description": "Search query or Zenodo record ID (for info/download)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to save downloaded files (default: ./data)"
        }
    }
    output_type = "string"

    def forward(self, action: str, query: str, output_dir: str = "./data") -> str:
        import requests

        if action == "search":
            return self._search(query)
        elif action == "info":
            return self._get_info(query)
        elif action == "download":
            return self._download(query, output_dir)
        else:
            return f"Unknown action: {action}. Use 'search', 'info', or 'download'"

    def _search(self, query: str) -> str:
        import requests

        params = {
            "q": f"{query} AND resource_type.type:dataset",
            "size": 10,
            "sort": "mostrecent"
        }

        try:
            response = requests.get(
                "https://zenodo.org/api/records",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                meta = hit.get("metadata", {})
                results.append({
                    "id": hit.get("id"),
                    "title": meta.get("title", "No title"),
                    "description": meta.get("description", "")[:200],
                    "creators": [c.get("name", "") for c in meta.get("creators", [])[:3]],
                    "doi": hit.get("doi"),
                    "url": hit.get("links", {}).get("html"),
                    "files_count": len(hit.get("files", []))
                })

            if not results:
                return f"No Zenodo datasets found for: {query}"

            formatted = f"Zenodo search results for: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. [{r['id']}] {r['title']}\n"
                formatted += f"   Creators: {', '.join(r['creators'])}\n"
                formatted += f"   DOI: {r['doi']}\n"
                formatted += f"   Files: {r['files_count']}\n"
                formatted += f"   URL: {r['url']}\n\n"

            print(f"Found {len(results)} datasets on Zenodo")
            return formatted

        except Exception as e:
            return f"Zenodo search error: {str(e)}"

    def _get_info(self, record_id: str) -> str:
        import requests

        try:
            response = requests.get(
                f"https://zenodo.org/api/records/{record_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            meta = data.get("metadata", {})
            files = data.get("files", [])

            info = {
                "id": data.get("id"),
                "doi": data.get("doi"),
                "title": meta.get("title"),
                "description": meta.get("description", "")[:500],
                "creators": [c.get("name") for c in meta.get("creators", [])],
                "keywords": meta.get("keywords", []),
                "publication_date": meta.get("publication_date"),
                "files": [
                    {
                        "key": f.get("key"),
                        "size_mb": round(f.get("size", 0) / 1024 / 1024, 2),
                        "checksum": f.get("checksum"),
                        "download_url": f.get("links", {}).get("self")
                    }
                    for f in files
                ]
            }

            return json.dumps(info, indent=2)

        except Exception as e:
            return f"Error getting Zenodo record info: {str(e)}"

    def _download(self, record_id: str, output_dir: str) -> str:
        import requests
        from tqdm import tqdm

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(
                f"https://zenodo.org/api/records/{record_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            files = data.get("files", [])
            downloaded = []

            for f in files:
                url = f.get("links", {}).get("self")
                filename = f.get("key")
                file_size = f.get("size", 0)

                if not url or not filename:
                    continue

                file_path = output_path / filename

                print(f"Downloading {filename} ({file_size/1024/1024:.1f} MB)...")

                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()

                with open(file_path, "wb") as out_file:
                    for chunk in resp.iter_content(chunk_size=8192):
                        out_file.write(chunk)

                downloaded.append({
                    "filename": filename,
                    "path": str(file_path),
                    "size_mb": round(file_size / 1024 / 1024, 2)
                })
                print(f"  Downloaded: {file_path}")

            return json.dumps({
                "record_id": record_id,
                "downloaded_files": downloaded,
                "output_directory": str(output_path)
            }, indent=2)

        except Exception as e:
            return f"Download error: {str(e)}"


class CERNOpenDataTool(Tool):
    """
    Tool for accessing CERN Open Data Portal.
    """
    name = "cern_opendata"
    description = """
    Accesses the CERN Open Data Portal for collision data and simulations.
    Provides access to:
    - CMS, ATLAS, LHCb, ALICE datasets
    - Monte Carlo simulations
    - Analysis examples and code
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Action: 'search', 'info', 'get_files'"
        },
        "query": {
            "type": "string",
            "description": "Search query or record ID"
        },
        "experiment": {
            "type": "string",
            "description": "Filter by experiment: 'CMS', 'ATLAS', 'LHCb', 'ALICE', 'all'"
        }
    }
    output_type = "string"

    def forward(self, action: str, query: str, experiment: str = "all") -> str:
        import requests

        base_url = "http://opendata.cern.ch/api/records"

        if action == "search":
            return self._search(query, experiment, base_url)
        elif action == "info":
            return self._get_info(query, base_url)
        elif action == "get_files":
            return self._get_files(query, base_url)
        else:
            return f"Unknown action: {action}"

    def _search(self, query: str, experiment: str, base_url: str) -> str:
        import requests

        params = {"q": query, "size": 10}
        if experiment != "all":
            params["q"] = f"{query} AND experiment:{experiment}"

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                meta = hit.get("metadata", {})
                results.append({
                    "recid": meta.get("recid"),
                    "title": meta.get("title", "No title"),
                    "experiment": meta.get("experiment", ["Unknown"])[0] if isinstance(meta.get("experiment"), list) else meta.get("experiment", "Unknown"),
                    "type": meta.get("type", {}).get("primary", "Unknown"),
                    "collision_type": meta.get("collision_information", {}).get("type", ""),
                    "energy": meta.get("collision_information", {}).get("energy", ""),
                })

            if not results:
                return f"No CERN Open Data records found for: {query}"

            formatted = f"CERN Open Data results for: {query}\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. [Record {r['recid']}] {r['title']}\n"
                formatted += f"   Experiment: {r['experiment']}\n"
                formatted += f"   Type: {r['type']}\n"
                if r['collision_type']:
                    formatted += f"   Collision: {r['collision_type']} at {r['energy']}\n"
                formatted += f"   URL: http://opendata.cern.ch/record/{r['recid']}\n\n"

            print(f"Found {len(results)} records on CERN Open Data")
            return formatted

        except Exception as e:
            return f"CERN Open Data search error: {str(e)}"

    def _get_info(self, record_id: str, base_url: str) -> str:
        import requests

        try:
            response = requests.get(
                f"http://opendata.cern.ch/api/records/{record_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            meta = data.get("metadata", {})

            info = {
                "recid": meta.get("recid"),
                "title": meta.get("title"),
                "experiment": meta.get("experiment"),
                "type": meta.get("type"),
                "collision_information": meta.get("collision_information"),
                "date_created": meta.get("date_created"),
                "date_published": meta.get("date_published"),
                "categories": meta.get("categories"),
                "abstract": meta.get("abstract", {}).get("description", "")[:500],
                "methodology": meta.get("methodology", {}).get("description", "")[:300],
                "n_files": len(meta.get("files", []))
            }

            return json.dumps(info, indent=2)

        except Exception as e:
            return f"Error getting record info: {str(e)}"

    def _get_files(self, record_id: str, base_url: str) -> str:
        import requests

        try:
            response = requests.get(
                f"http://opendata.cern.ch/api/records/{record_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            files = data.get("metadata", {}).get("files", [])

            file_list = []
            for f in files:
                file_list.append({
                    "filename": f.get("key", f.get("filename", "unknown")),
                    "size_mb": round(f.get("size", 0) / 1024 / 1024, 2),
                    "uri": f.get("uri"),
                    "checksum": f.get("checksum")
                })

            return json.dumps({
                "record_id": record_id,
                "total_files": len(file_list),
                "files": file_list
            }, indent=2)

        except Exception as e:
            return f"Error getting files: {str(e)}"


class ROOTFileLoaderTool(Tool):
    """
    Tool for loading and inspecting ROOT files.
    """
    name = "root_loader"
    description = """
    Loads and inspects ROOT files using uproot.
    Can:
    - List contents (trees, histograms) of ROOT files
    - Read specific branches from TTrees
    - Load histograms
    - Get file metadata
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Action: 'list', 'read_tree', 'read_histogram', 'info'"
        },
        "file_path": {
            "type": "string",
            "description": "Path to ROOT file (local or URL)"
        },
        "object_name": {
            "type": "string",
            "description": "Name of tree/histogram to read (for read_tree/read_histogram)"
        },
        "branches": {
            "type": "string",
            "description": "Comma-separated branch names to read (for read_tree). Use 'all' for all branches"
        }
    }
    output_type = "string"

    def forward(self, action: str, file_path: str, object_name: str = "", branches: str = "all") -> str:
        try:
            import uproot
        except ImportError:
            return "Error: uproot not installed. Install with: pip install uproot"

        if action == "list":
            return self._list_contents(file_path)
        elif action == "read_tree":
            return self._read_tree(file_path, object_name, branches)
        elif action == "read_histogram":
            return self._read_histogram(file_path, object_name)
        elif action == "info":
            return self._get_info(file_path)
        else:
            return f"Unknown action: {action}"

    def _list_contents(self, file_path: str) -> str:
        import uproot

        try:
            with uproot.open(file_path) as f:
                contents = {
                    "file": file_path,
                    "objects": []
                }

                for key, obj in f.items():
                    obj_info = {
                        "name": key,
                        "classname": obj.classname if hasattr(obj, 'classname') else type(obj).__name__
                    }

                    if hasattr(obj, 'keys'):  # TTree
                        obj_info["type"] = "TTree"
                        obj_info["branches"] = list(obj.keys())[:20]  # First 20 branches
                        obj_info["n_branches"] = len(obj.keys())
                        obj_info["n_entries"] = obj.num_entries

                    elif hasattr(obj, 'values'):  # Histogram
                        obj_info["type"] = "TH1/TH2"
                        obj_info["n_bins"] = len(obj.values())

                    contents["objects"].append(obj_info)

                return json.dumps(contents, indent=2)

        except Exception as e:
            return f"Error reading ROOT file: {str(e)}"

    def _read_tree(self, file_path: str, tree_name: str, branches: str) -> str:
        import uproot
        import numpy as np

        try:
            with uproot.open(file_path) as f:
                tree = f[tree_name]

                if branches == "all":
                    branch_list = list(tree.keys())[:10]  # Limit to 10 branches
                else:
                    branch_list = [b.strip() for b in branches.split(",")]

                # Read limited number of events for preview
                arrays = tree.arrays(branch_list, library="np", entry_stop=1000)

                result = {
                    "tree_name": tree_name,
                    "total_entries": tree.num_entries,
                    "entries_read": min(1000, tree.num_entries),
                    "branches": {}
                }

                for name, arr in arrays.items():
                    result["branches"][name] = {
                        "dtype": str(arr.dtype),
                        "shape": list(arr.shape),
                        "min": float(np.min(arr)) if np.issubdtype(arr.dtype, np.number) else None,
                        "max": float(np.max(arr)) if np.issubdtype(arr.dtype, np.number) else None,
                        "mean": float(np.mean(arr)) if np.issubdtype(arr.dtype, np.number) else None,
                        "sample_values": arr[:5].tolist() if len(arr) > 0 else []
                    }

                print(f"Read {result['entries_read']} entries from {tree_name}")
                return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error reading tree: {str(e)}"

    def _read_histogram(self, file_path: str, hist_name: str) -> str:
        import uproot
        import numpy as np

        try:
            with uproot.open(file_path) as f:
                hist = f[hist_name]

                values = hist.values()
                edges = hist.axis().edges()

                result = {
                    "histogram_name": hist_name,
                    "n_bins": len(values),
                    "total_entries": float(np.sum(values)),
                    "mean": float(np.average(edges[:-1] + np.diff(edges)/2, weights=values)),
                    "x_range": [float(edges[0]), float(edges[-1])],
                    "bin_edges": edges.tolist(),
                    "bin_values": values.tolist()
                }

                if hasattr(hist, 'variances'):
                    result["bin_errors"] = np.sqrt(hist.variances()).tolist()

                print(f"Read histogram {hist_name} with {result['n_bins']} bins")
                return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error reading histogram: {str(e)}"

    def _get_info(self, file_path: str) -> str:
        import uproot
        import os

        try:
            with uproot.open(file_path) as f:
                info = {
                    "file_path": file_path,
                    "file_size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2) if os.path.exists(file_path) else None,
                    "n_objects": len(f.keys()),
                    "object_names": list(f.keys())
                }

                return json.dumps(info, indent=2)

        except Exception as e:
            return f"Error getting file info: {str(e)}"


class DatasetDownloaderTool(Tool):
    """
    General-purpose dataset downloader for HEP data.
    """
    name = "dataset_downloader"
    description = """
    Downloads datasets from various sources:
    - Direct URLs
    - Zenodo records
    - CERN Open Data
    - XRootD paths

    Supports resumable downloads and checksum verification.
    """
    inputs = {
        "url": {
            "type": "string",
            "description": "URL or path to download from"
        },
        "output_path": {
            "type": "string",
            "description": "Local path to save the file"
        },
        "checksum": {
            "type": "string",
            "description": "Optional MD5 checksum for verification"
        }
    }
    output_type = "string"

    def forward(self, url: str, output_path: str, checksum: str = "") -> str:
        import requests
        import hashlib
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists and matches checksum
        if output_file.exists() and checksum:
            existing_checksum = self._compute_checksum(output_file)
            if existing_checksum == checksum:
                return json.dumps({
                    "status": "exists",
                    "message": "File already exists with matching checksum",
                    "path": str(output_file)
                })

        try:
            # Handle different URL types
            if url.startswith("root://"):
                return self._download_xrootd(url, output_path)

            # HTTP/HTTPS download
            print(f"Downloading from {url}...")

            response = requests.get(url, stream=True, timeout=3600)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                            print(f"  Progress: {progress:.1f}%")

            # Verify checksum if provided
            if checksum:
                actual_checksum = self._compute_checksum(output_file)
                if actual_checksum != checksum:
                    return json.dumps({
                        "status": "error",
                        "message": f"Checksum mismatch. Expected: {checksum}, Got: {actual_checksum}",
                        "path": str(output_file)
                    })

            file_size = output_file.stat().st_size / 1024 / 1024

            return json.dumps({
                "status": "success",
                "path": str(output_file),
                "size_mb": round(file_size, 2),
                "checksum_verified": bool(checksum)
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "url": url
            })

    def _compute_checksum(self, file_path: Path) -> str:
        import hashlib

        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def _download_xrootd(self, url: str, output_path: str) -> str:
        """Download using XRootD protocol"""
        try:
            # Try using xrdcp command
            import subprocess
            result = subprocess.run(
                ["xrdcp", url, output_path],
                capture_output=True,
                text=True,
                timeout=3600
            )
            if result.returncode == 0:
                return json.dumps({
                    "status": "success",
                    "path": output_path,
                    "method": "xrootd"
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": result.stderr
                })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"XRootD download failed: {str(e)}"
            })
