"""Validated text tables, file readers, and an opt-in HTML stripper."""

from copernicus.data.html import strip_html
from copernicus.data.readers import read_frame
from copernicus.data.table import TextTable

__all__ = ["TextTable", "read_frame", "strip_html"]
