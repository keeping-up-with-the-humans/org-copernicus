"""Formulas of the OLS estimators: one that cannot be parsed, or whose design holds no numbers.

A formula that ``formulaic`` cannot parse, or that has several parts, is a setting the library
refuses. A formula whose design holds a value that is no real number (a text, a complex number,
a duration, a list) does not fit the data, and neither does one whose evaluation fails (a
column of dates). In these cases the error is the library's own, and where ``formulaic``,
Python or numpy raised an error, that error is its cause. A design of real numbers that are
not floats, such as decimals or fractions, is evaluated.
"""

from __future__ import annotations

import decimal
import fractions

import formulaic
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from formula_support import N_TEXTS, formula_data, formula_frame
from formulaic.errors import FormulaParsingError, FormulaSyntaxError

from copernicus.corrections import ols_outcome_ppi, ols_predictor_ppi
from copernicus.errors import ConfigurationError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

UNPARSEABLE = (
    pytest.param("1 +* log_chars", FormulaSyntaxError, id="two operators"),
    pytest.param("1 + log_chars +", FormulaSyntaxError, id="open end"),
    pytest.param("1 + (log_chars", FormulaSyntaxError, id="open parenthesis"),
    pytest.param("1 + log_chars # note", FormulaSyntaxError, id="comment outside a call"),
    pytest.param("5", FormulaSyntaxError, id="number alone"),
    pytest.param("1 + .", FormulaParsingError, id="dot"),
    pytest.param("1 + center(log_chars +)", SyntaxError, id="python in a call"),
    pytest.param("1 + {1: 2}", SyntaxError, id="python in braces"),
    pytest.param("1 + center(\ud800)", ValueError, id="character without an encoding"),
)
# Formulas whose design holds a value that is no real number: the formula, the name of the
# design column and the type of the value. TEXT is passed on by formulaic 1.2.2 as it is.
TEXT = (
    ("1 + label", "label", "str"),
    ("1 + digits", "digits", "str"),
    ("1 + raw", "raw", "bytes"),
    ('1 + I("a")', "I('a')", "str"),
    ('0 + I("1.5")', "I('1.5')", "str"),
)
NOT_REAL = (
    ("1 + wave", "wave", "complex"),
    ("1 + center(wave)", "center(wave)", "complex"),
    ("1 + center(log_chars + 1j)", "center(log_chars + 1j)", "complex"),
    ("1 + span", "span", "Timedelta"),
    ("1 + pairs", "pairs", "ndarray"),
    ("1 + I(bytes_like.tolist())", "I(bytes_like.tolist())", "bytearray"),
    ("1 + I(wrapped.tolist())", "I(wrapped.tolist())", "ndarray"),
    ("1 + I({'v': mixed})", "I({'v': mixed})[v]", "str"),
    ("1 + I(odd.tolist())", "I(odd.tolist())", "complex128"),
    ("1 + I({'v': odd})", "I({'v': odd})[v]", "complex128"),
    ("1 + I({'v': timed})", "I({'v': timed})[v]", "timedelta64"),
    ("1 + I({'v': foreign})", "I({'v': foreign})[v]", "_Foreign"),
)
# Formulas whose evaluation fails inside the formula library with an error of the class named.
UNEVALUABLE = (
    ("1 + when", TypeError),
    ("1 + center(exact)", TypeError),
    ("1 + I(np.array(2))", IndexError),
    ("1 + C(kind, contrasts=np.array([-1, 1]))", IndexError),
)
REAL = ("small", "unsigned", "single", "flag", "nullable_float", "nullable_int", "nullable_flag")
REAL_OBJECTS = ("decimals", "fractions", "integers", "short_floats", "short_integers")


class _Foreign:
    """A number of another package: it turns into a float and is no real number of Python."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value

    def __mul__(self, other: float) -> _Foreign:
        return _Foreign(self.value * other)

    __rmul__ = __mul__


def _typed_frame() -> pd.DataFrame:
    """The frame with columns of types that give no real numbers and of types that do.

    ``when`` holds dates, ``span`` durations, ``exact`` decimal objects and ``wave`` complex
    numbers; ``label`` and ``digits`` are text columns of the ``string`` type of pandas, the
    second with texts that read as numbers, ``raw`` holds the same texts as bytes in an Arrow
    column, and ``kind`` is a text column of two levels. The columns of ``REAL`` hold
    integers, floats and booleans of several widths, some of the nullable types of pandas.

    The other columns hold objects over ``quarters``, the covariate rounded to quarters. Those
    of ``REAL_OBJECTS`` hold real numbers: decimals, fractions, Python's integers and numpy's
    32-bit floats and 8-bit integers. ``bytes_like`` holds the quarters as texts in mutable
    bytes, ``wrapped`` as texts in arrays without a dimension, ``mixed`` a float followed by
    texts, ``odd`` decimals in turn with complex numbers of numpy, ``timed`` decimals in turn
    with durations of numpy, ``foreign`` numbers of a class of this module, and ``pairs`` is
    an Arrow column of lists of two floats.
    """
    frame = formula_frame()
    values = frame["log_chars"].to_numpy(dtype=float)
    order = np.arange(N_TEXTS)
    quarters = np.round(values * 4) / 4
    texts = [str(value) for value in quarters]
    numbers_in_turn = [
        decimal.Decimal(text) if index % 2 else np.complex128(complex(float(text), 1.0))
        for index, text in enumerate(texts)
    ]
    objects = {
        "decimals": [decimal.Decimal(text) for text in texts],
        "fractions": [fractions.Fraction(float(value)) for value in quarters],
        "integers": [int(value) for value in np.round(quarters)],
        "short_floats": [np.float32(value) for value in quarters],
        "short_integers": [np.int8(value) for value in np.round(quarters)],
        "timed": [
            decimal.Decimal(text) if index % 2 else np.timedelta64(index, "D")
            for index, text in enumerate(texts)
        ],
        "foreign": [_Foreign(float(value)) for value in quarters],
        "bytes_like": [bytearray(text, "ascii") for text in texts],
        "wrapped": [np.array(text, dtype=object) for text in texts],
        "mixed": [float(quarters[0]), *texts[1:]],
        "odd": numbers_in_turn,
    }
    return frame.assign(
        quarters=quarters,
        pairs=pd.array(
            [[float(value), float(value) + 1.0] for value in quarters],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
        ),
        **{name: pd.Series(held, dtype=object) for name, held in objects.items()},
        when=pd.Timestamp("2026-01-01") + pd.to_timedelta(order, unit="D"),
        span=pd.to_timedelta(order, unit="h"),
        exact=[decimal.Decimal(repr(round(float(value), 3))) for value in values],
        wave=values + 1j * values,
        label=pd.array(np.where(values > 0, "a", "b"), dtype="string"),
        digits=pd.array([repr(round(float(value), 2)) for value in values], dtype="string"),
        raw=pd.array(
            [repr(round(float(value), 2)).encode() for value in values],
            dtype=pd.ArrowDtype(pa.binary()),
        ),
        kind=np.where(values > 0, "a", "b"),
        small=order.astype("int8"),
        unsigned=order.astype("uint16"),
        single=values.astype("float32"),
        flag=values > 0,
        nullable_float=pd.array(values, dtype="Float64"),
        nullable_int=pd.array(order, dtype="Int64"),
        nullable_flag=pd.array(values > 0, dtype="boolean"),
    )


@pytest.mark.parametrize(("formula", "cause"), UNPARSEABLE)
def test_a_formula_that_cannot_be_parsed_is_refused_as_a_setting(
    formula: str, cause: type[Exception]
) -> None:
    data = formula_data()
    with pytest.raises(SettingsError, match="cannot be parsed") as refusal:
        ols_outcome_ppi(data, formula)
    message = str(refusal.value)
    assert isinstance(refusal.value.__cause__, cause)
    assert message.startswith(f"formula {formula!r} cannot be parsed")
    assert str(refusal.value.__cause__).splitlines()[0] in message
    assert "\n" not in message
    assert "\x1b" not in message
    with pytest.raises(SettingsError, match="^covariates .* cannot be parsed") as refusal:
        ols_predictor_ppi(data, "y", covariates=formula)
    assert isinstance(refusal.value.__cause__, cause)


@pytest.mark.parametrize(
    "formula", ["1 + log_chars | y", "1 + log_chars | 1 + y", "1 + log_chars | missing_col"]
)
def test_a_formula_of_several_parts_is_refused_as_a_setting(formula: str) -> None:
    data = formula_data()
    with pytest.raises(SettingsError, match="^formula must be one right-hand side") as refusal:
        ols_outcome_ppi(data, formula)
    assert repr(formula) in str(refusal.value)
    with pytest.raises(SettingsError, match="^covariates must be one right-hand side"):
        ols_predictor_ppi(data, "y", covariates=formula)


@pytest.mark.parametrize(("formula", "cause"), UNEVALUABLE)
def test_a_formula_whose_evaluation_fails_in_the_formula_library_is_refused(
    formula: str, cause: type[Exception]
) -> None:
    data = formula_data(_typed_frame())
    with pytest.raises(ConfigurationError, match="cannot be evaluated on the data") as refusal:
        ols_outcome_ppi(data, formula)
    assert isinstance(refusal.value.__cause__, cause)
    assert str(refusal.value.__cause__) in str(refusal.value)
    with pytest.raises(ConfigurationError, match="^covariates .* cannot be evaluated") as refusal:
        ols_predictor_ppi(data, "y", covariates=formula)
    assert isinstance(refusal.value.__cause__, cause)


def test_contrasts_of_the_right_form_are_evaluated() -> None:
    data = formula_data(_typed_frame())
    estimate = ols_outcome_ppi(data, "1 + C(kind, contrasts=np.array([[-1], [1]]))")
    assert np.isfinite(estimate.point)


@pytest.mark.parametrize(("formula", "column", "kind"), TEXT + NOT_REAL)
def test_a_design_column_with_a_value_that_is_no_real_number_is_refused_and_named(
    formula: str, column: str, kind: str
) -> None:
    frame = _typed_frame()
    if (formula, column, kind) in TEXT:
        design = formulaic.model_matrix(formula, frame, context={})
        if all(dtype.kind in "biuf" for dtype in design.dtypes):
            pytest.skip("the installed formulaic codes this text column by its levels")
    data = formula_data(frame)
    with pytest.raises(ConfigurationError, match="that are not real numbers") as refusal:
        ols_outcome_ppi(data, formula)
    assert f"{column} ({kind} in " in str(refusal.value)
    assert str(refusal.value).startswith(f"formula {formula!r} gives design column(s)")
    assert refusal.value.__cause__ is None
    with pytest.raises(ConfigurationError, match="^covariates .* that are not real numbers"):
        ols_predictor_ppi(data, "y", covariates=formula)


@pytest.mark.parametrize("column", REAL_OBJECTS)
def test_real_numbers_held_as_objects_give_the_design_built_by_hand(column: str) -> None:
    frame = _typed_frame()
    data = formula_data(frame)
    held = np.array([float(value) for value in frame[column]])
    by_hand = ols_outcome_ppi(data, X=np.column_stack([np.ones(N_TEXTS), held]), coefficient=1)
    estimate = ols_outcome_ppi(data, f"1 + I({{'v': {column}}})", coefficient=1)
    assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
    assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)
    predictor_by_hand = ols_predictor_ppi(data, "y", X=held[:, None], coefficient=2)
    predictor = ols_predictor_ppi(data, "y", covariates=f"I({{'v': {column}}})", coefficient=2)
    assert predictor.point == pytest.approx(predictor_by_hand.point, rel=1e-12)
    assert predictor.se == pytest.approx(predictor_by_hand.se, rel=1e-12)


def test_a_bar_inside_a_python_expression_does_not_part_a_formula() -> None:
    data = formula_data()
    either = ((data.frame["log_chars"] > 0) | (data.frame["log_chars"] < -1)).to_numpy(dtype=float)
    by_hand = ols_outcome_ppi(data, X=np.column_stack([np.ones(N_TEXTS), either]), coefficient=1)
    formula = "1 + I((log_chars > 0) | (log_chars < -1))"
    estimate = ols_outcome_ppi(data, formula, coefficient=1)
    assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
    assert np.isfinite(ols_predictor_ppi(data, "y", covariates=formula).point)


def test_a_column_label_that_is_a_missing_value_does_not_disturb_a_formula() -> None:
    frame = formula_frame()
    plain = ols_outcome_ppi(formula_data(frame), "1 + log_chars")
    frame.columns = pd.Index(frame.columns, dtype=object)
    frame[pd.NA] = np.arange(N_TEXTS)
    data = formula_data(frame)
    assert ols_outcome_ppi(data, "1 + log_chars").point == plain.point
    assert np.isfinite(ols_predictor_ppi(data, "y", covariates="1 + log_chars").point)
    with pytest.raises(ConfigurationError, match=r"names column\(s\) 'nope' that the data lack"):
        ols_outcome_ppi(data, "1 + nope")


@pytest.mark.parametrize(
    ("held", "cause"),
    [
        ([10**399 + index for index in range(N_TEXTS)], OverflowError),
        (
            [decimal.Decimal("sNaN"), *[decimal.Decimal(index) for index in range(1, N_TEXTS)]],
            decimal.InvalidOperation,
        ),
    ],
    ids=["an integer of 400 digits", "a signalling decimal"],
)
def test_a_number_that_cannot_become_a_float_is_refused_with_its_cause(
    held: list[object], cause: type[Exception]
) -> None:
    data = formula_data(formula_frame().assign(odd_number=pd.Series(held, dtype=object)))
    with pytest.raises(ConfigurationError, match="^formula .* odd_number") as refusal:
        ols_outcome_ppi(data, "1 + I({'v': odd_number})")
    assert isinstance(refusal.value.__cause__, cause)
    with pytest.raises(ConfigurationError, match="^covariates ") as refusal:
        ols_predictor_ppi(data, "y", covariates="I({'v': odd_number})")
    assert isinstance(refusal.value.__cause__, cause)


class _Unconvertible(decimal.Decimal):
    """A decimal that passes arithmetic unchanged and signals when it is turned into a float."""

    def __mul__(self, other: object) -> _Unconvertible:
        return self

    __rmul__ = __mul__

    def __float__(self) -> float:
        raise decimal.InvalidOperation("no float")


def test_a_decimal_that_signals_in_the_last_conversion_is_refused_with_its_cause() -> None:
    held = pd.Series([_Unconvertible(index) for index in range(N_TEXTS)], dtype=object)
    data = formula_data(formula_frame().assign(odd_number=held))
    with pytest.raises(ConfigurationError, match="cannot be turned into floats") as refusal:
        ols_outcome_ppi(data, "1 + I({'v': odd_number})")
    assert isinstance(refusal.value.__cause__, decimal.InvalidOperation)


def test_a_design_with_an_infinite_value_is_refused() -> None:
    data = formula_data()
    finite = "does not give a finite value in every column for every text"
    with pytest.raises(ConfigurationError, match=finite) as refusal:
        ols_outcome_ppi(data, "1 + I(log_chars * np.inf)")
    assert str(refusal.value).startswith("formula '1 + I(log_chars * np.inf)' does not give")
    assert refusal.value.__cause__ is None
    with pytest.raises(ConfigurationError, match=f"^covariates .* {finite}"):
        ols_predictor_ppi(data, "y", covariates="I(log_chars * np.inf)")


@pytest.mark.parametrize("arrow", [pa.decimal128(12, 2), pa.decimal256(40, 2)], ids=str)
def test_a_column_of_decimals_gives_the_design_built_by_hand(arrow: pa.DataType) -> None:
    """Decimals arrive in the design as objects; they are numbers and are taken as such."""
    frame = formula_frame()
    quarters = np.round(frame["log_chars"].to_numpy(dtype=float) * 4) / 4
    amounts = [decimal.Decimal(str(value)) for value in quarters]
    data = formula_data(frame.assign(amount=pd.array(amounts, dtype=pd.ArrowDtype(arrow))))
    by_hand = ols_outcome_ppi(data, X=np.column_stack([np.ones(N_TEXTS), quarters]))
    for formula in ("1 + amount", "1 + I(amount)"):
        estimate = ols_outcome_ppi(data, formula, coefficient=1)
        assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
        assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)
    predictor_by_hand = ols_predictor_ppi(data, "y", X=quarters[:, None], coefficient=2)
    predictor = ols_predictor_ppi(data, "y", covariates="amount", coefficient=2)
    assert predictor.point == pytest.approx(predictor_by_hand.point, rel=1e-12)
    assert predictor.se == pytest.approx(predictor_by_hand.se, rel=1e-12)


def test_a_formula_that_reads_a_name_that_two_columns_have_is_refused() -> None:
    frame = formula_frame()
    twice = frame[["log_chars", "log_chars"]].set_axis(["extra", "extra"], axis=1)
    data = formula_data(pd.concat([frame, twice], axis=1))
    named = "reads 'extra', which pandas does not resolve to one column of the data"
    with pytest.raises(ConfigurationError, match=f"^formula '1 \\+ extra' {named}") as refusal:
        ols_outcome_ppi(data, "1 + extra")
    assert refusal.value.__cause__ is None
    with pytest.raises(ConfigurationError, match=f"^covariates 'extra' {named}"):
        ols_predictor_ppi(data, "y", covariates="extra")
    assert np.isfinite(ols_outcome_ppi(data, "1 + log_chars").point)


@pytest.mark.parametrize("column", REAL)
def test_a_column_of_real_numbers_of_any_width_gives_the_design_built_by_hand(column: str) -> None:
    frame = _typed_frame()
    data = formula_data(frame)
    design = np.column_stack([np.ones(N_TEXTS), frame[column].to_numpy(dtype=float)])
    by_hand = ols_outcome_ppi(data, X=design, coefficient=1)
    estimate = ols_outcome_ppi(data, f"1 + {column}", coefficient=1)
    assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
    assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)


def test_a_text_column_of_the_string_type_is_coded_by_its_levels_when_the_formula_asks() -> None:
    frame = _typed_frame()
    data = formula_data(frame)
    second_level = (frame["label"] == "b").to_numpy(dtype=float)
    by_hand = ols_outcome_ppi(data, X=np.column_stack([np.ones(N_TEXTS), second_level]))
    estimate = ols_outcome_ppi(data, "1 + C(label)", coefficient=1)
    assert estimate.details["columns"] == ("Intercept", "C(label)[T.b]")
    assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
    assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)
