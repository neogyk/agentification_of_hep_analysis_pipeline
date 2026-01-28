"""
Tests for Data Fetchers

Tests for CERN Open Data Portal and Zenodo fetchers.
Includes both unit tests (with mocking) and integration tests (live API calls).
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import responses

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.data_fetchers import (
    CERNOpenDataFetcher,
    ZenodoFetcher,
    FileInfo,
    DatasetInfo,
    DownloadResult,
    search_cern_opendata,
    search_zenodo,
    download_from_cern,
    download_from_zenodo,
    create_session,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cern_fetcher():
    """Create a CERN Open Data fetcher instance."""
    return CERNOpenDataFetcher(timeout=10)


@pytest.fixture
def zenodo_fetcher():
    """Create a Zenodo fetcher instance."""
    return ZenodoFetcher(timeout=10)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for downloads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_cern_search_response():
    """Mock response for CERN Open Data search."""
    return {
        "hits": {
            "hits": [
                {
                    "metadata": {
                        "recid": 12345,
                        "title": "CMS Open Data Sample",
                        "experiment": ["CMS"],
                        "abstract": {"description": "Test dataset for unit tests"},
                        "collision_information": {
                            "energy": "13TeV",
                            "type": "pp"
                        },
                        "date_published": "2023-01-01",
                        "keywords": ["physics", "cms", "open data"],
                        "files": [
                            {
                                "key": "test_file.root",
                                "size": 1024000,
                                "checksum": "md5:abc123",
                                "uri": "/record/12345/files/test_file.root"
                            }
                        ]
                    }
                }
            ],
            "total": 1
        }
    }


@pytest.fixture
def mock_zenodo_search_response():
    """Mock response for Zenodo search."""
    return {
        "hits": {
            "hits": [
                {
                    "id": 67890,
                    "doi": "10.5281/zenodo.67890",
                    "metadata": {
                        "title": "HEP Analysis Dataset",
                        "description": "Test dataset for Zenodo",
                        "creators": [{"name": "Test Author"}],
                        "publication_date": "2023-06-15",
                        "keywords": ["hep", "physics"]
                    },
                    "links": {
                        "html": "https://zenodo.org/record/67890"
                    },
                    "files": [
                        {
                            "key": "data.csv",
                            "size": 512000,
                            "checksum": "md5:def456",
                            "links": {
                                "self": "https://zenodo.org/api/files/abc/data.csv"
                            }
                        }
                    ]
                }
            ],
            "total": 1
        }
    }


# =============================================================================
# Unit Tests - Data Classes
# =============================================================================

class TestFileInfo:
    """Tests for FileInfo data class."""

    def test_file_info_creation(self):
        """Test creating a FileInfo object."""
        file_info = FileInfo(
            filename="test.root",
            size_bytes=1048576,  # 1 MB
            checksum="md5:abc123"
        )

        assert file_info.filename == "test.root"
        assert file_info.size_bytes == 1048576
        assert file_info.size_mb == 1.0
        assert file_info.checksum == "md5:abc123"

    def test_file_info_to_dict(self):
        """Test converting FileInfo to dictionary."""
        file_info = FileInfo(
            filename="test.root",
            size_bytes=2097152,  # 2 MB
            download_url="https://example.com/test.root"
        )

        result = file_info.to_dict()

        assert result["filename"] == "test.root"
        assert result["size_mb"] == 2.0
        assert result["download_url"] == "https://example.com/test.root"


class TestDatasetInfo:
    """Tests for DatasetInfo data class."""

    def test_dataset_info_creation(self):
        """Test creating a DatasetInfo object."""
        dataset = DatasetInfo(
            id="12345",
            title="Test Dataset",
            description="A test dataset",
            creators=["Author 1", "Author 2"],
            experiment="CMS"
        )

        assert dataset.id == "12345"
        assert dataset.title == "Test Dataset"
        assert len(dataset.creators) == 2
        assert dataset.experiment == "CMS"

    def test_dataset_info_with_files(self):
        """Test DatasetInfo with files."""
        files = [
            FileInfo("file1.root", 1048576),
            FileInfo("file2.root", 2097152)
        ]

        dataset = DatasetInfo(
            id="12345",
            title="Test Dataset",
            files=files
        )

        result = dataset.to_dict()

        assert result["n_files"] == 2
        assert result["total_size_mb"] == 3.0


class TestDownloadResult:
    """Tests for DownloadResult data class."""

    def test_successful_download_result(self):
        """Test successful download result."""
        result = DownloadResult(
            success=True,
            filepath="/path/to/file.root",
            size_bytes=1048576,
            checksum_verified=True
        )

        assert result.success is True
        assert result.filepath == "/path/to/file.root"
        assert result.to_dict()["size_mb"] == 1.0

    def test_failed_download_result(self):
        """Test failed download result."""
        result = DownloadResult(
            success=False,
            error_message="Connection timeout"
        )

        assert result.success is False
        assert result.error_message == "Connection timeout"


# =============================================================================
# Unit Tests - CERN Open Data Fetcher (Mocked)
# =============================================================================

class TestCERNOpenDataFetcherMocked:
    """Unit tests for CERN Open Data fetcher with mocked responses."""

    @responses.activate
    def test_search_success(self, cern_fetcher, mock_cern_search_response):
        """Test successful search."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records",
            json=mock_cern_search_response,
            status=200
        )

        results = cern_fetcher.search("CMS muon")

        assert len(results) == 1
        assert results[0].id == "12345"
        assert results[0].title == "CMS Open Data Sample"
        assert results[0].experiment == "CMS"
        assert len(results[0].files) == 1

    @responses.activate
    def test_search_with_experiment_filter(self, cern_fetcher, mock_cern_search_response):
        """Test search with experiment filter."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records",
            json=mock_cern_search_response,
            status=200
        )

        results = cern_fetcher.search("muon", experiment="CMS")

        assert len(results) == 1
        # Verify the query included experiment filter
        assert "experiment:CMS" in responses.calls[0].request.url

    @responses.activate
    def test_search_empty_results(self, cern_fetcher):
        """Test search with no results."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records",
            json={"hits": {"hits": [], "total": 0}},
            status=200
        )

        results = cern_fetcher.search("nonexistent_dataset_xyz")

        assert len(results) == 0

    @responses.activate
    def test_search_api_error(self, cern_fetcher):
        """Test search with API error."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records",
            json={"error": "Server error"},
            status=500
        )

        with pytest.raises(Exception):
            cern_fetcher.search("test")

    @responses.activate
    def test_get_record(self, cern_fetcher):
        """Test getting a specific record."""
        mock_response = {
            "metadata": {
                "recid": 12345,
                "title": "Detailed Record",
                "experiment": ["CMS"],
                "abstract": {"description": "Full description here"},
                "collision_information": {"energy": "13TeV", "type": "pp"},
                "doi": "10.1234/example",
                "files": [
                    {"key": "file1.root", "size": 1000000, "uri": "/files/file1.root"},
                    {"key": "file2.root", "size": 2000000, "uri": "/files/file2.root"}
                ]
            }
        }

        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records/12345",
            json=mock_response,
            status=200
        )

        dataset = cern_fetcher.get_record("12345")

        assert dataset.id == "12345"
        assert dataset.title == "Detailed Record"
        assert len(dataset.files) == 2
        assert dataset.doi == "10.1234/example"

    @responses.activate
    def test_get_file_list(self, cern_fetcher):
        """Test getting file list for a record."""
        mock_response = {
            "metadata": {
                "recid": 12345,
                "title": "Test",
                "files": [
                    {"key": "file1.root", "size": 1000000},
                    {"key": "file2.root", "size": 2000000},
                    {"key": "file3.root", "size": 3000000}
                ]
            }
        }

        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records/12345",
            json=mock_response,
            status=200
        )

        files = cern_fetcher.get_file_list("12345")

        assert len(files) == 3
        assert files[0].filename == "file1.root"
        assert files[2].size_bytes == 3000000


# =============================================================================
# Unit Tests - Zenodo Fetcher (Mocked)
# =============================================================================

class TestZenodoFetcherMocked:
    """Unit tests for Zenodo fetcher with mocked responses."""

    @responses.activate
    def test_search_success(self, zenodo_fetcher, mock_zenodo_search_response):
        """Test successful search."""
        responses.add(
            responses.GET,
            "https://zenodo.org/api/records",
            json=mock_zenodo_search_response,
            status=200
        )

        results = zenodo_fetcher.search("HEP analysis")

        assert len(results) == 1
        assert results[0].id == "67890"
        assert results[0].title == "HEP Analysis Dataset"
        assert results[0].doi == "10.5281/zenodo.67890"
        assert len(results[0].files) == 1

    @responses.activate
    def test_search_with_resource_type(self, zenodo_fetcher, mock_zenodo_search_response):
        """Test search with resource type filter."""
        responses.add(
            responses.GET,
            "https://zenodo.org/api/records",
            json=mock_zenodo_search_response,
            status=200
        )

        results = zenodo_fetcher.search("physics", resource_type="software")

        assert len(results) == 1
        assert "resource_type.type:software" in responses.calls[0].request.url

    @responses.activate
    def test_search_empty_results(self, zenodo_fetcher):
        """Test search with no results."""
        responses.add(
            responses.GET,
            "https://zenodo.org/api/records",
            json={"hits": {"hits": [], "total": 0}},
            status=200
        )

        results = zenodo_fetcher.search("nonexistent_xyz_123")

        assert len(results) == 0

    @responses.activate
    def test_get_record(self, zenodo_fetcher):
        """Test getting a specific record."""
        mock_response = {
            "id": 67890,
            "doi": "10.5281/zenodo.67890",
            "metadata": {
                "title": "Detailed Zenodo Record",
                "description": "Full description",
                "creators": [{"name": "Author One"}, {"name": "Author Two"}],
                "publication_date": "2023-06-15",
                "keywords": ["hep", "physics", "data"]
            },
            "links": {"html": "https://zenodo.org/record/67890"},
            "files": [
                {
                    "key": "data.h5",
                    "size": 5000000,
                    "checksum": "md5:xyz789",
                    "links": {"self": "https://zenodo.org/api/files/xyz/data.h5"}
                }
            ]
        }

        responses.add(
            responses.GET,
            "https://zenodo.org/api/records/67890",
            json=mock_response,
            status=200
        )

        dataset = zenodo_fetcher.get_record("67890")

        assert dataset.id == "67890"
        assert dataset.title == "Detailed Zenodo Record"
        assert len(dataset.creators) == 2
        assert len(dataset.files) == 1

    @responses.activate
    def test_download_file(self, zenodo_fetcher, temp_dir):
        """Test downloading a file."""
        file_content = b"Test file content for download test"

        responses.add(
            responses.GET,
            "https://zenodo.org/api/files/test/data.txt",
            body=file_content,
            status=200,
            headers={"content-length": str(len(file_content))}
        )

        file_info = FileInfo(
            filename="data.txt",
            size_bytes=len(file_content),
            download_url="https://zenodo.org/api/files/test/data.txt"
        )

        result = zenodo_fetcher.download_file(
            file_info,
            temp_dir,
            verify_checksum=False
        )

        assert result.success is True
        assert result.filepath == str(Path(temp_dir) / "data.txt")

        # Verify file content
        with open(result.filepath, 'rb') as f:
            assert f.read() == file_content


# =============================================================================
# Unit Tests - Convenience Functions (Mocked)
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @responses.activate
    def test_search_cern_opendata_function(self, mock_cern_search_response):
        """Test the search_cern_opendata convenience function."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records",
            json=mock_cern_search_response,
            status=200
        )

        results = search_cern_opendata("CMS", experiment="CMS", max_results=5)

        assert len(results) == 1
        assert results[0]["id"] == "12345"
        assert "files" in results[0]

    @responses.activate
    def test_search_zenodo_function(self, mock_zenodo_search_response):
        """Test the search_zenodo convenience function."""
        responses.add(
            responses.GET,
            "https://zenodo.org/api/records",
            json=mock_zenodo_search_response,
            status=200
        )

        results = search_zenodo("physics", resource_type="dataset", max_results=10)

        assert len(results) == 1
        assert results[0]["id"] == "67890"


# =============================================================================
# Integration Tests (Live API - Marked for Optional Running)
# =============================================================================

@pytest.mark.integration
class TestCERNOpenDataIntegration:
    """
    Integration tests for CERN Open Data Portal.

    These tests make real API calls and should be run separately.
    Run with: pytest -m integration
    """

    def test_live_search(self, cern_fetcher):
        """Test live search against CERN Open Data."""
        results = cern_fetcher.search("CMS", experiment="CMS", max_results=3)

        assert len(results) > 0
        assert all(r.experiment == "CMS" for r in results if r.experiment)

    def test_live_get_record(self, cern_fetcher):
        """Test getting a specific real record."""
        # Record 12020 is a well-known CMS dataset
        try:
            dataset = cern_fetcher.get_record("12020")
            assert dataset.id == "12020"
            assert dataset.experiment == "CMS"
        except Exception:
            pytest.skip("Could not fetch record - API may be unavailable")

    def test_live_search_with_energy_filter(self, cern_fetcher):
        """Test search with collision energy filter."""
        results = cern_fetcher.search(
            "muon",
            experiment="CMS",
            collision_energy="13TeV",
            max_results=5
        )

        # Just verify we get results back
        assert isinstance(results, list)


@pytest.mark.integration
class TestZenodoIntegration:
    """
    Integration tests for Zenodo.

    These tests make real API calls and should be run separately.
    Run with: pytest -m integration
    """

    def test_live_search(self, zenodo_fetcher):
        """Test live search against Zenodo."""
        results = zenodo_fetcher.search(
            "particle physics",
            resource_type="dataset",
            max_results=3
        )

        assert len(results) > 0

    def test_live_search_software(self, zenodo_fetcher):
        """Test searching for software."""
        results = zenodo_fetcher.search(
            "ROOT CERN",
            resource_type="software",
            max_results=3
        )

        assert isinstance(results, list)


# =============================================================================
# Test Session Creation
# =============================================================================

class TestSessionCreation:
    """Tests for HTTP session creation."""

    def test_create_session_default(self):
        """Test creating session with default parameters."""
        session = create_session()

        assert session is not None
        assert hasattr(session, 'get')

    def test_create_session_custom_retries(self):
        """Test creating session with custom retry settings."""
        session = create_session(retries=5, backoff_factor=1.0)

        assert session is not None


# =============================================================================
# Test Checksum Verification
# =============================================================================

class TestChecksumVerification:
    """Tests for checksum verification functionality."""

    def test_compute_checksum_md5(self, temp_dir, cern_fetcher):
        """Test MD5 checksum computation."""
        # Create a test file
        test_file = Path(temp_dir) / "test_checksum.txt"
        test_content = b"Hello, World!"
        with open(test_file, 'wb') as f:
            f.write(test_content)

        checksum = cern_fetcher._compute_checksum(test_file, "md5")

        # Known MD5 of "Hello, World!"
        assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"

    def test_compute_checksum_sha256(self, temp_dir, cern_fetcher):
        """Test SHA256 checksum computation."""
        test_file = Path(temp_dir) / "test_checksum.txt"
        test_content = b"Hello, World!"
        with open(test_file, 'wb') as f:
            f.write(test_content)

        checksum = cern_fetcher._compute_checksum(test_file, "sha256")

        # Known SHA256 of "Hello, World!"
        assert checksum == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @responses.activate
    def test_timeout_handling(self):
        """Test handling of timeout errors."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records",
            body=Exception("Connection timeout")
        )

        fetcher = CERNOpenDataFetcher(timeout=1)

        with pytest.raises(Exception):
            fetcher.search("test")

    @responses.activate
    def test_invalid_record_id(self, cern_fetcher):
        """Test handling of invalid record ID."""
        responses.add(
            responses.GET,
            "http://opendata.cern.ch/api/records/99999999",
            json={"error": "Record not found"},
            status=404
        )

        with pytest.raises(Exception):
            cern_fetcher.get_record("99999999")

    def test_download_without_url(self, cern_fetcher, temp_dir):
        """Test download with missing URL."""
        file_info = FileInfo(
            filename="test.root",
            size_bytes=1000
            # No download_url or uri
        )

        result = cern_fetcher.download_file(file_info, temp_dir)

        assert result.success is False
        assert "No download URL" in result.error_message


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run unit tests only by default
    pytest.main([__file__, "-v", "-m", "not integration"])
