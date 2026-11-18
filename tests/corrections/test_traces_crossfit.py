"""The cross-fitted prediction records the inner folds whose design has a numerical rank below 3.

The rank is the one that ``numpy.linalg.lstsq`` with ``rcond=None`` returns. On the folds of these
data the design is rank-deficient in exact arithmetic, so the line ``gold ~ 1 + score + covariate``
has many least-squares solutions; the library keeps the one that ``lstsq`` returns, of smallest
norm, counts the fold and logs one warning per call. The assertions pin the
results for fixed test inputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from traces_support import (
    CROSSFIT_K,
    CROSSFIT_SEED,
    close,
    crossfit_data,
    crossfit_frame,
    rank_deficient_crossfit_frame,
    study_corpus,
)

from copernicus.corrections import pseudo_values, repeated_study
from copernicus.corrections.families._crossfit_ghat import crossfit_ghat

pytestmark = pytest.mark.filterwarnings("error")

CROSSFIT = "copernicus.corrections.families._crossfit_ghat"
STUDY_SIZE, STUDY_DRAWS, STUDY_SEED = 40, 6, 1


def crossfit_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The warnings that the cross-fitted prediction logged."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == CROSSFIT and record.levelno == logging.WARNING
    ]


def test_a_fold_whose_design_is_rank_deficient_is_counted_and_warned_about_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = crossfit_data(rank_deficient_crossfit_frame())
    with caplog.at_level(logging.WARNING):
        values = pseudo_values(data, rectifier="crossfit", k=CROSSFIT_K, seed=CROSSFIT_SEED)
    assert values.n_rank_deficient_folds == 1
    (message,) = crossfit_warnings(caplog)
    assert message.startswith(
        "cross-fitted prediction: the design of intercept, score and covariate has a numerical "
        "rank below 3, as numpy.linalg.lstsq with rcond=None returns it, on the labelled training "
        "rows of 1 of the 2 folds (fold 0: rank 2)"
    )
    assert "returns the coefficients of smallest norm of what remains" in message
    assert "When the design is rank-deficient in exact arithmetic, many lines fit" in message
    assert "n_rank_deficient_folds" in message
    # The fit keeps the minimum-norm solution.
    assert (values.mean_plugin, values.mean_rectified) == close(
        (5.235690345975788, 6.55289170500898)
    )
    assert values.lambda_star == close(-0.159435973603069)
    assert float(values.ghat.iloc[0]) == close(7.032364894675667)
    assert float(values.ghat.iloc[-1]) == close(4.6604649216555725)


def test_the_cross_fitted_prediction_names_its_rank_deficient_folds() -> None:
    deficient = crossfit_ghat(
        crossfit_data(rank_deficient_crossfit_frame()), k=CROSSFIT_K, seed=CROSSFIT_SEED
    )
    ordinary = crossfit_ghat(crossfit_data(crossfit_frame()), k=CROSSFIT_K, seed=CROSSFIT_SEED)
    assert deficient.rank_deficient_folds == (0,)
    assert ordinary.rank_deficient_folds == ()
    assert deficient.columns == ordinary.columns == ("noise", "noise")
    assert np.isfinite(deficient.ghat).all() and deficient.ghat.shape == (32,)


def test_without_a_rank_deficient_fold_the_count_is_zero_and_nothing_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        values = pseudo_values(
            crossfit_data(crossfit_frame()), rectifier="crossfit", k=CROSSFIT_K, seed=CROSSFIT_SEED
        )
    assert values.n_rank_deficient_folds == 0
    assert crossfit_warnings(caplog) == []
    assert (values.mean_plugin, values.mean_rectified) == close(
        (4.9699028755673575, 6.317008099090186)
    )


@pytest.mark.parametrize("rectifier", ["identity", "calibrated"])
def test_the_rectifiers_without_folds_count_none(
    rectifier: str, caplog: pytest.LogCaptureFixture
) -> None:
    data = crossfit_data(rank_deficient_crossfit_frame())
    with caplog.at_level(logging.WARNING):
        values = pseudo_values(data, rectifier=rectifier)  # type: ignore[arg-type]
    assert values.n_rank_deficient_folds == 0
    assert crossfit_warnings(caplog) == []


def test_the_study_counts_the_rank_deficient_folds_of_its_cross_fitted_estimator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The covariate is a line in the only score column, so every one of the five inner folds of
    # every draw has a design of rank 2.
    with caplog.at_level(logging.WARNING):
        table = repeated_study(
            study_corpus(covariate_on_score=True),
            truth_gold="truth",
            sizes=(STUDY_SIZE,),
            draws=STUDY_DRAWS,
            seed=STUDY_SEED,
        )
    assert list(table.columns[-2:]) == ["n_rank_deficient_folds", "n_se_exactly_zero"]
    rows = table.set_index(["estimator", "estimand"])
    for estimand in ("mean", "long_short_difference"):
        row = rows.loc[("ppi_crossfit", estimand)]
        assert (row["n_draws"], row["n_undefined"]) == (STUDY_DRAWS, 0)
        assert row["n_rank_deficient_folds"] == 5 * STUDY_DRAWS
    others = table[table["estimator"] != "ppi_crossfit"]
    assert (others["n_rank_deficient_folds"] == 0).all() and len(others) == 6
    # The study sums up the warnings of the cross-fit: one for each draw, whose column serves
    # both estimands.
    assert crossfit_warnings(caplog) == []
    (summary,) = [
        r.getMessage() for r in caplog.records if r.name == "copernicus.corrections.study"
    ]
    assert summary.startswith(f"repeated study, size {STUDY_SIZE}: ") and ", 6 say that " in summary
    assert rows.loc[("ppi_crossfit", "mean"), "estimate_mean"] == close(1.9686429986871292)


def test_a_study_without_rank_deficient_folds_counts_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        table = repeated_study(
            study_corpus(covariate_on_score=False),
            truth_gold="truth",
            sizes=(STUDY_SIZE,),
            draws=STUDY_DRAWS,
            seed=STUDY_SEED,
        )
    assert (table["n_rank_deficient_folds"] == 0).all()
    assert str(table["n_rank_deficient_folds"].dtype) == "int64"
    assert crossfit_warnings(caplog) == []
