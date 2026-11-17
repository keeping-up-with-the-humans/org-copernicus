"""Validated text tables, file readers, an opt-in HTML stripper, and the whitespace rule."""

from copernicus.data.html import strip_html
from copernicus.data.readers import read_frame
from copernicus.data.table import TextTable, collapse_whitespace

__all__ = ["TextTable", "collapse_whitespace", "read_frame", "strip_html"]
