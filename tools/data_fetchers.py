"""
Data Fetchers for HEP Analysis

Dedicated tools for fetching data from:
- CERN Open Data Portal
- Zenodo Repository

These are standalone functions that can be used directly or wrapped as smolagents Tools.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FileInfo:
    """Information about a downloadable file."""
    filename: str
    size_bytes: int
    checksum: Optional[str] = None
    checksum_type: str = "md5"
    download_url: Optional[str] = None
    uri: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_mb,
            "checksum": self.checksum,
            "checksum_type": self.checksum_type,
            "download_url": self.download_url,
            "uri": self.uri
        }


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    id: str
    title: str
    description: str = ""
    creators: List[str] = field(default_factory=list)
    experiment: Optional[str] = None
    collision_energy: Optional[str] = None
    collision_type: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    publication_date: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    files: List[FileInfo] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "creators": self.creators,
            "experiment": self.experiment,
            "collision_energy": self.collision_energy,
            "collision_type": self.collision_type,
            "doi": self.doi,
            "url": self.url,
            "publication_date": self.publication_date,
            "keywords": self.keywords,
            "files": [f.to_dict() for f in self.files],
            "n_files": len(self.files),
            "total_size_mb": sum(f.size_mb for f in self.files)
        }


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    filepath: Optional[str] = None
    size_bytes: int = 0
    checksum_verified: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "filepath": self.filepath,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2) if self.size_bytes else 0,
            "checksum_verified": self.checksum_verified,
            "error_message": self.error_message
        }


class DataSource(Enum):
    """Supported data sources."""
    CERN_OPEN_DATA = "cern_opendata"
    ZENODO = "zenodo"


# =============================================================================
# HTTP Session with Retry Logic
# =============================================================================

def create_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =============================================================================
# CERN Open Data Portal Fetcher
# =============================================================================

class CERNOpenDataFetcher:
    """
    Fetcher for CERN Open Data Portal.

    The CERN Open Data Portal provides access to datasets from LHC experiments
    including CMS, ATLAS, LHCb, and ALICE.

    API Documentation: http://opendata.cern.ch/docs/api
    """

    BASE_URL = "http://opendata.cern.ch/api/records"
    PORTAL_URL = "http://opendata.cern.ch"

    def __init__(self, timeout: int = 30):
        """
        Initialize the CERN Open Data fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = create_session()

    def search(
        self,
        query: str,
        experiment: Optional[str] = None,
        data_type: Optional[str] = None,
        collision_energy: Optional[str] = None,
        max_results: int = 10
    ) -> List[DatasetInfo]:
        """
        Search for datasets on CERN Open Data Portal.

        Args:
            query: Search query string
            experiment: Filter by experiment (CMS, ATLAS, LHCb, ALICE)
            data_type: Filter by data type (Dataset, Software, Documentation)
            collision_energy: Filter by collision energy (e.g., "13TeV")
            max_results: Maximum number of results to return

        Returns:
            List of DatasetInfo objects
        """
        # Build query
        q_parts = [query]
        if experiment:
            q_parts.append(f"experiment:{experiment}")
        if data_type:
            q_parts.append(f"type.primary:{data_type}")
        if collision_energy:
            q_parts.append(f'collision_energy:"{collision_energy}"')

        params = {
            "q": " AND ".join(q_parts),
            "size": max_results
        }

        logger.info(f"Searching CERN Open Data: {params['q']}")

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                meta = hit.get("metadata", {})

                # Extract experiment (can be list or string)
                exp = meta.get("experiment", [])
                if isinstance(exp, list):
                    exp = exp[0] if exp else None

                # Extract collision info
                collision_info = meta.get("collision_information", {})

                dataset = DatasetInfo(
                    id=str(meta.get("recid", "")),
                    title=meta.get("title", "Untitled"),
                    description=meta.get("abstract", {}).get("description", "")[:500],
                    experiment=exp,
                    collision_energy=collision_info.get("energy"),
                    collision_type=collision_info.get("type"),
                    url=f"{self.PORTAL_URL}/record/{meta.get('recid')}",
                    publication_date=meta.get("date_published"),
                    keywords=meta.get("keywords", [])
                )

                # Extract file info
                for f in meta.get("files", []):
                    dataset.files.append(FileInfo(
                        filename=f.get("key", f.get("filename", "unknown")),
                        size_bytes=f.get("size", 0),
                        checksum=f.get("checksum"),
                        uri=f.get("uri")
                    ))

                results.append(dataset)

            logger.info(f"Found {len(results)} datasets")
            return results

        except requests.RequestException as e:
            logger.error(f"Search failed: {e}")
            raise

    def get_record(self, record_id: str) -> DatasetInfo:
        """
        Get detailed information about a specific record.

        Args:
            record_id: CERN Open Data record ID

        Returns:
            DatasetInfo object with full details
        """
        url = f"{self.BASE_URL}/{record_id}"

        logger.info(f"Fetching record: {record_id}")

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            meta = data.get("metadata", {})

            # Extract experiment
            exp = meta.get("experiment", [])
            if isinstance(exp, list):
                exp = exp[0] if exp else None

            # Extract collision info
            collision_info = meta.get("collision_information", {})

            dataset = DatasetInfo(
                id=str(meta.get("recid", record_id)),
                title=meta.get("title", "Untitled"),
                description=meta.get("abstract", {}).get("description", ""),
                experiment=exp,
                collision_energy=collision_info.get("energy"),
                collision_type=collision_info.get("type"),
                doi=meta.get("doi"),
                url=f"{self.PORTAL_URL}/record/{record_id}",
                publication_date=meta.get("date_published"),
                keywords=meta.get("keywords", [])
            )

            # Extract all files
            for f in meta.get("files", []):
                dataset.files.append(FileInfo(
                    filename=f.get("key", f.get("filename", "unknown")),
                    size_bytes=f.get("size", 0),
                    checksum=f.get("checksum"),
                    uri=f.get("uri")
                ))

            return dataset

        except requests.RequestException as e:
            logger.error(f"Failed to fetch record {record_id}: {e}")
            raise

    def get_file_list(self, record_id: str) -> List[FileInfo]:
        """
        Get list of files for a record.

        Args:
            record_id: CERN Open Data record ID

        Returns:
            List of FileInfo objects
        """
        dataset = self.get_record(record_id)
        return dataset.files

    def download_file(
        self,
        file_info: FileInfo,
        output_dir: Union[str, Path],
        verify_checksum: bool = True,
        progress_callback: Optional[callable] = None
    ) -> DownloadResult:
        """
        Download a file from CERN Open Data.

        Args:
            file_info: FileInfo object with download details
            output_dir: Directory to save the file
            verify_checksum: Whether to verify the checksum after download
            progress_callback: Optional callback for progress updates

        Returns:
            DownloadResult object
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / file_info.filename

        # Construct download URL
        if file_info.uri:
            # CERN Open Data uses EOS URIs, construct HTTP URL
            download_url = f"{self.PORTAL_URL}{file_info.uri}"
        elif file_info.download_url:
            download_url = file_info.download_url
        else:
            return DownloadResult(
                success=False,
                error_message="No download URL available"
            )

        logger.info(f"Downloading {file_info.filename} ({file_info.size_mb} MB)")

        try:
            response = self.session.get(download_url, stream=True, timeout=3600)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', file_info.size_bytes))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

            # Verify checksum
            checksum_ok = True
            if verify_checksum and file_info.checksum:
                actual_checksum = self._compute_checksum(filepath, file_info.checksum_type)
                expected = file_info.checksum
                if ":" in expected:
                    expected = expected.split(":")[1]
                checksum_ok = actual_checksum == expected

                if not checksum_ok:
                    logger.warning(f"Checksum mismatch for {file_info.filename}")

            return DownloadResult(
                success=True,
                filepath=str(filepath),
                size_bytes=filepath.stat().st_size,
                checksum_verified=checksum_ok
            )

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return DownloadResult(
                success=False,
                error_message=str(e)
            )

    def _compute_checksum(self, filepath: Path, checksum_type: str = "md5") -> str:
        """Compute file checksum."""
        if checksum_type == "md5":
            hasher = hashlib.md5()
        elif checksum_type == "sha256":
            hasher = hashlib.sha256()
        else:
            hasher = hashlib.md5()

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)

        return hasher.hexdigest()


# =============================================================================
# Zenodo Fetcher
# =============================================================================

class ZenodoFetcher:
    """
    Fetcher for Zenodo repository.

    Zenodo is a general-purpose open repository that hosts datasets,
    software, and publications from all fields of science.

    API Documentation: https://developers.zenodo.org/
    """

    BASE_URL = "https://zenodo.org/api/records"

    def __init__(self, access_token: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Zenodo fetcher.

        Args:
            access_token: Optional Zenodo access token for authenticated requests
            timeout: Request timeout in seconds
        """
        self.access_token = access_token or os.getenv("ZENODO_TOKEN")
        self.timeout = timeout
        self.session = create_session()

        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"

    def search(
        self,
        query: str,
        communities: Optional[List[str]] = None,
        resource_type: str = "dataset",
        sort: str = "mostrecent",
        max_results: int = 10
    ) -> List[DatasetInfo]:
        """
        Search for records on Zenodo.

        Args:
            query: Search query string
            communities: Filter by community (e.g., ["cern", "hep"])
            resource_type: Filter by resource type (dataset, software, publication)
            sort: Sort order (mostrecent, bestmatch)
            max_results: Maximum number of results

        Returns:
            List of DatasetInfo objects
        """
        # Build query
        q_parts = [query]
        if resource_type:
            q_parts.append(f"resource_type.type:{resource_type}")

        params = {
            "q": " AND ".join(q_parts),
            "size": max_results,
            "sort": sort
        }

        if communities:
            params["communities"] = ",".join(communities)

        logger.info(f"Searching Zenodo: {params['q']}")

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                meta = hit.get("metadata", {})

                dataset = DatasetInfo(
                    id=str(hit.get("id", "")),
                    title=meta.get("title", "Untitled"),
                    description=meta.get("description", "")[:500],
                    creators=[c.get("name", "") for c in meta.get("creators", [])],
                    doi=hit.get("doi"),
                    url=hit.get("links", {}).get("html"),
                    publication_date=meta.get("publication_date"),
                    keywords=meta.get("keywords", [])
                )

                # Extract file info
                for f in hit.get("files", []):
                    dataset.files.append(FileInfo(
                        filename=f.get("key", "unknown"),
                        size_bytes=f.get("size", 0),
                        checksum=f.get("checksum"),
                        download_url=f.get("links", {}).get("self")
                    ))

                results.append(dataset)

            logger.info(f"Found {len(results)} records")
            return results

        except requests.RequestException as e:
            logger.error(f"Search failed: {e}")
            raise

    def get_record(self, record_id: str) -> DatasetInfo:
        """
        Get detailed information about a specific record.

        Args:
            record_id: Zenodo record ID

        Returns:
            DatasetInfo object with full details
        """
        url = f"{self.BASE_URL}/{record_id}"

        logger.info(f"Fetching Zenodo record: {record_id}")

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            meta = data.get("metadata", {})

            dataset = DatasetInfo(
                id=str(data.get("id", record_id)),
                title=meta.get("title", "Untitled"),
                description=meta.get("description", ""),
                creators=[c.get("name", "") for c in meta.get("creators", [])],
                doi=data.get("doi"),
                url=data.get("links", {}).get("html"),
                publication_date=meta.get("publication_date"),
                keywords=meta.get("keywords", [])
            )

            # Extract all files
            for f in data.get("files", []):
                dataset.files.append(FileInfo(
                    filename=f.get("key", "unknown"),
                    size_bytes=f.get("size", 0),
                    checksum=f.get("checksum"),
                    download_url=f.get("links", {}).get("self")
                ))

            return dataset

        except requests.RequestException as e:
            logger.error(f"Failed to fetch record {record_id}: {e}")
            raise

    def get_file_list(self, record_id: str) -> List[FileInfo]:
        """
        Get list of files for a record.

        Args:
            record_id: Zenodo record ID

        Returns:
            List of FileInfo objects
        """
        dataset = self.get_record(record_id)
        return dataset.files

    def download_file(
        self,
        file_info: FileInfo,
        output_dir: Union[str, Path],
        verify_checksum: bool = True,
        progress_callback: Optional[callable] = None
    ) -> DownloadResult:
        """
        Download a file from Zenodo.

        Args:
            file_info: FileInfo object with download details
            output_dir: Directory to save the file
            verify_checksum: Whether to verify the checksum after download
            progress_callback: Optional callback for progress updates

        Returns:
            DownloadResult object
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / file_info.filename

        if not file_info.download_url:
            return DownloadResult(
                success=False,
                error_message="No download URL available"
            )

        logger.info(f"Downloading {file_info.filename} ({file_info.size_mb} MB)")

        try:
            response = self.session.get(
                file_info.download_url,
                stream=True,
                timeout=3600
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', file_info.size_bytes))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

            # Verify checksum
            checksum_ok = True
            if verify_checksum and file_info.checksum:
                actual_checksum = self._compute_checksum(filepath)
                expected = file_info.checksum
                if ":" in expected:
                    expected = expected.split(":")[1]
                checksum_ok = actual_checksum == expected

                if not checksum_ok:
                    logger.warning(f"Checksum mismatch for {file_info.filename}")

            return DownloadResult(
                success=True,
                filepath=str(filepath),
                size_bytes=filepath.stat().st_size,
                checksum_verified=checksum_ok
            )

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return DownloadResult(
                success=False,
                error_message=str(e)
            )

    def download_record(
        self,
        record_id: str,
        output_dir: Union[str, Path],
        file_filter: Optional[callable] = None
    ) -> List[DownloadResult]:
        """
        Download all files from a record.

        Args:
            record_id: Zenodo record ID
            output_dir: Directory to save files
            file_filter: Optional function to filter files (returns True to download)

        Returns:
            List of DownloadResult objects
        """
        files = self.get_file_list(record_id)

        if file_filter:
            files = [f for f in files if file_filter(f)]

        results = []
        for file_info in files:
            result = self.download_file(file_info, output_dir)
            results.append(result)

        return results

    def _compute_checksum(self, filepath: Path) -> str:
        """Compute MD5 checksum of a file."""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()


# =============================================================================
# Convenience Functions
# =============================================================================

def search_cern_opendata(
    query: str,
    experiment: Optional[str] = None,
    max_results: int = 10
) -> List[Dict]:
    """
    Search CERN Open Data Portal.

    Args:
        query: Search query
        experiment: Filter by experiment (CMS, ATLAS, LHCb, ALICE)
        max_results: Maximum results

    Returns:
        List of dataset dictionaries
    """
    fetcher = CERNOpenDataFetcher()
    results = fetcher.search(query, experiment=experiment, max_results=max_results)
    return [r.to_dict() for r in results]


def search_zenodo(
    query: str,
    resource_type: str = "dataset",
    max_results: int = 10
) -> List[Dict]:
    """
    Search Zenodo repository.

    Args:
        query: Search query
        resource_type: Type of resource (dataset, software, publication)
        max_results: Maximum results

    Returns:
        List of dataset dictionaries
    """
    fetcher = ZenodoFetcher()
    results = fetcher.search(query, resource_type=resource_type, max_results=max_results)
    return [r.to_dict() for r in results]


def download_from_cern(
    record_id: str,
    output_dir: str,
    filename: Optional[str] = None
) -> Dict:
    """
    Download files from CERN Open Data.

    Args:
        record_id: Record ID
        output_dir: Output directory
        filename: Specific filename to download (downloads all if None)

    Returns:
        Download result dictionary
    """
    fetcher = CERNOpenDataFetcher()
    files = fetcher.get_file_list(record_id)

    if filename:
        files = [f for f in files if f.filename == filename]

    results = []
    for file_info in files:
        result = fetcher.download_file(file_info, output_dir)
        results.append(result.to_dict())

    return {
        "record_id": record_id,
        "output_dir": output_dir,
        "files_downloaded": len([r for r in results if r["success"]]),
        "results": results
    }


def download_from_zenodo(
    record_id: str,
    output_dir: str,
    filename: Optional[str] = None
) -> Dict:
    """
    Download files from Zenodo.

    Args:
        record_id: Record ID
        output_dir: Output directory
        filename: Specific filename to download (downloads all if None)

    Returns:
        Download result dictionary
    """
    fetcher = ZenodoFetcher()
    files = fetcher.get_file_list(record_id)

    if filename:
        files = [f for f in files if f.filename == filename]

    results = []
    for file_info in files:
        result = fetcher.download_file(file_info, output_dir)
        results.append(result.to_dict())

    return {
        "record_id": record_id,
        "output_dir": output_dir,
        "files_downloaded": len([r for r in results if r["success"]]),
        "results": results
    }
