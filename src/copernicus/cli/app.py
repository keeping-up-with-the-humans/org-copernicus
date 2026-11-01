"""The ``copernicus`` command-line application."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from copernicus.cli._baseline import baseline_app
from copernicus.cli._code import code_app
from copernicus.cli._corrections import corrections_app
from copernicus.cli._embeddings import embed_app, hybrid_app, similarity_app
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors
from copernicus.cli._references import references_app
from copernicus.cli._store import store_app
from copernicus.cli._validity import validity_app
from copernicus.rubric import load_rubric

logger = logging.getLogger(__name__)

__all__ = ["app"]

app = typer.Typer(
    name="copernicus",
    help="Code constructs from text with rubric-based LLM coders.",
    no_args_is_help=True,
)
rubric_app = typer.Typer(help="Check rubric files.", no_args_is_help=True)


@rubric_app.command("validate")
def validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Rubric JSON file.")],
    verbose: Verbose = False,
) -> None:
    """Check a rubric file against the rubric schema and the id and marker-name rules."""
    with cli_logging(verbose), cli_errors():
        rubric = load_rubric(path)
        logger.info("validated rubric %s from %s", rubric.rubric_id, path)
        n_markers = sum(len(attribute.markers) for attribute in rubric.attributes)
        typer.echo(
            f"valid: {rubric.rubric_id} ({len(rubric.attributes)} attributes, {n_markers} markers)"
        )


app.add_typer(rubric_app, name="rubric")
app.add_typer(code_app, name="code")
app.add_typer(store_app, name="store")
app.add_typer(references_app, name="references")
app.add_typer(embed_app, name="embed")
app.add_typer(similarity_app, name="similarity")
app.add_typer(hybrid_app, name="hybrid")
app.add_typer(validity_app, name="validity")
app.add_typer(baseline_app, name="baseline")
app.add_typer(corrections_app, name="corrections")
