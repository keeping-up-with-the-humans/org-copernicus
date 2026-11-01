"""The ``store`` commands: merge shard stores and export tables."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors
from copernicus.cli._output import check_table_output, write_table
from copernicus.coders import (
    export_construct_scores,
    export_evidence,
    export_provenance,
    export_scores,
    export_text_scores,
    merge_stores,
)
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["store_app"]

store_app = typer.Typer(help="Merge and export run stores.", no_args_is_help=True)


def _provenance_frame(store_path: Path) -> pd.DataFrame:
    """Return the run information of a store as a ``key``, ``value`` table."""
    info = export_provenance(store_path)
    return pd.DataFrame({"key": list(info), "value": list(info.values())})


EXPORTS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "scores": export_scores,
    "construct": export_construct_scores,
    "text": export_text_scores,
    "evidence": export_evidence,
    "provenance": _provenance_frame,
}


@store_app.command("merge")
def merge(
    sources: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Shard stores of one run."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Store that receives the rows.")],
    verbose: Verbose = False,
) -> None:
    """Merge shard stores of one run into a single store."""
    with cli_logging(verbose), cli_errors():
        counts = merge_stores(sources, out)
        logger.info("merged %d stores into %s", len(sources), out)
        for name, count in counts.items():
            typer.echo(f"{name}: {count}")


@store_app.command("export")
def export(
    store: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Run store.")],
    what: Annotated[str, typer.Option("--what", help=f"One of {', '.join(EXPORTS)}.")],
    out: Annotated[Path, typer.Option("--out", help="Output file, .csv or .parquet.")],
    verbose: Verbose = False,
) -> None:
    """Write one table of a run store to a CSV or parquet file."""
    with cli_logging(verbose), cli_errors():
        if what not in EXPORTS:
            raise SettingsError(f"--what must be one of {', '.join(EXPORTS)}, got {what!r}")
        check_table_output(out)
        frame = EXPORTS[what](store)
        write_table(frame, out)
        logger.info("exported %s of %s to %s", what, store, out)
        typer.echo(f"wrote {len(frame)} rows to {out}")
