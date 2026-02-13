"""Google Drive integration for fetching images."""

import io
import re
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Regex patterns for extracting file ID from Google Drive URLs
GDRIVE_URL_PATTERNS = [
    r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    r"drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
]


class GoogleDriveClient:
    """Client for downloading images from Google Drive."""

    def __init__(self, credentials_path: Path):
        """Initialize the Google Drive client.

        Args:
            credentials_path: Path to the service account credentials JSON file.
        """
        self.credentials_path = credentials_path
        self._service = None

    def _get_service(self):
        """Get or create the Google Drive service."""
        if self._service is None:
            credentials = Credentials.from_service_account_file(
                str(self.credentials_path),
                scopes=SCOPES
            )
            self._service = build("drive", "v3", credentials=credentials)
        return self._service

    @staticmethod
    def extract_file_id(url: str) -> str | None:
        """Extract file ID from a Google Drive URL.

        Args:
            url: Google Drive URL in various formats.

        Returns:
            File ID if found, None otherwise.
        """
        for pattern in GDRIVE_URL_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def find_gdrive_urls(text: str) -> list[str]:
        """Find all Google Drive URLs in text.

        Args:
            text: Text that may contain Google Drive URLs.

        Returns:
            List of Google Drive URLs found.
        """
        url_pattern = r"https?://drive\.google\.com/[^\s<>\"']+"
        return re.findall(url_pattern, text)

    def download_image(self, file_id: str) -> bytes:
        """Download an image from Google Drive.

        Args:
            file_id: Google Drive file ID.

        Returns:
            Image binary data.

        Raises:
            Exception: If download fails.
        """
        service = self._get_service()
        request = service.files().get_media(fileId=file_id)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        buffer.seek(0)
        return buffer.read()

    def download_image_from_url(self, url: str) -> bytes:
        """Download an image from a Google Drive URL.

        Args:
            url: Google Drive URL.

        Returns:
            Image binary data.

        Raises:
            ValueError: If URL is not a valid Google Drive URL.
            Exception: If download fails.
        """
        file_id = self.extract_file_id(url)
        if not file_id:
            raise ValueError(f"Invalid Google Drive URL: {url}")
        return self.download_image(file_id)
