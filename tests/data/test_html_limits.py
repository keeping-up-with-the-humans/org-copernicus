"""The limits of strip_html: plain text that it removes, and markup that it leaves.

The rule reads tags by their form, not by what the writer meant. The strings are made up, and the
expected outputs are the helper's, literals that make a later change of the rule visible.
"""

from __future__ import annotations

import sys

import pytest

from copernicus.data import strip_html

# The known limit of the rule: plain text between angle brackets is removed when it has the form
# of a tag that the rule accepts. One case or more per class: an element name, alone or with a
# word that may stand as an attribute without a value, also by Python's Unicode rules; a word with
# a hyphen, the form of a custom element name; two words joined by a colon, the form of a
# namespaced name. The other way round, real markup with an attribute without a value that the
# rule's list does not hold stays text.
KNOWN_LIMIT: tuple[tuple[str, str, str], ...] = (
    ("element_names_as_grades", "grade <A> or <B>", "grade or"),
    ("element_name_with_a_valueless_attribute", "<i open> to it", "to it"),
    ("element_name_as_a_placeholder", "insert <link> here", "insert here"),
    ("element_name_as_a_word", "at <time> noon", "at noon"),
    ("element_names_by_unicode_case", "Keep <ſ> and <İ> here.", "Keep and here."),
    ("custom_element_form", "The <well-known> issue remains.", "The issue remains."),
    ("namespaced_form", "Read <Note:important> first.", "Read first."),
    ("unlisted_valueless_attribute", "<div data-flag>x</div>", "<div data-flag>x"),
    ("unlisted_valueless_attribute_of_an_image", '<img alt="cat" lazy>', '<img alt="cat" lazy>'),
)

# The svg and math step: a scan selects the outermost complete spans. A self-closing <svg/> or
# <math/>, or a start tag with an attribute the step does not accept, also inside a comment,
# clears the elements open at that place; an outer element that is never completed is not
# selected, but a complete span nested in it is, before or after a reset. Inside a span only the
# tag forms that the step accepts go; the rest is left to the general rule, which keeps a name
# that is not on its list and has neither a hyphen nor a colon.
FOREIGN_ELEMENT_SCAN: tuple[tuple[str, str, str], ...] = (
    ("self_closing_svg_inside", "<svg><svg/><text>Hello</text></svg>", "<text>Hello</text>"),
    (
        "self_closing_svg_inside_a_comment",
        "<svg><!--<svg/>--><text>Hello</text></svg>",
        "<text>Hello</text>",
    ),
    ("svg_never_closed", '<svg><use href="#i">x</use>', '<use href="#i">x</use>'),
    ("complete_math_inside_an_unclosed_svg", "<svg><math><text>Hello</text></math>", "Hello"),
    (
        "complete_math_before_a_reset",
        "<svg><math><text>Hello</text></math><svg/><text>World</text></svg>",
        "Hello <text>World</text>",
    ),
    (
        "complete_math_after_a_reset",
        "<svg><svg/><math><text>Hello</text></math></svg>",
        "Hello",
    ),
    ("self_closing_math_inside", "<svg><math/><text>Hello</text></svg>", "<text>Hello</text>"),
    (
        "rejected_start_tag_inside",
        "<svg><svg data-flag><text>Hello</text></svg>",
        "<svg data-flag><text>Hello</text>",
    ),
    ("svg_followed_by_another_character", "<svg><svg!><text>Hello</text></svg>", "Hello"),
    ("name_without_an_ascii_letter", "<svg><é>Hello</é></svg>", "<é>Hello</é>"),
    ("equals_sign_without_a_value", "<svg><text bad=>Hello</text></svg>", "<text bad=>Hello"),
    (
        "svg_with_an_unlisted_valueless_attribute",
        '<svg data-flag><use href="#i">x</use></svg>',
        '<svg data-flag><use href="#i">x</use>',
    ),
)

# A doctype, a processing instruction and a CDATA section are not read as tags: their delimiters
# stay, and their contents are read by the rules like any text.
DECLARATIONS_WITH_TAGS_INSIDE: tuple[tuple[str, str, str], ...] = (
    ("cdata_section", "<![CDATA[<b>word</b>]]>", "<![CDATA[ word ]]>"),
    ("processing_instruction", '<?pi text="<b>word</b>"?>', '<?pi text=" word "?>'),
)

# The conditions of the rules for line breaks, blocks and images: they read lower-case names, a
# <br> without attributes and an alt attribute in lower case with its value in double quotes. A
# tag that they do not match is left to the general rule, which removes it and leaves a space
# only if it accepts the tag's form; a lower-case image tag that the image rule matches and that
# has no such alt becomes nothing, without a space.
FORMATTING_RULE_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("upper_case_line_break", "A<BR>B", "A B"),
    ("line_break_with_an_attribute", 'A<br class="x">B', "A B"),
    ("upper_case_paragraph", "A<P>B</P>C", "A B C"),
    ("single_quoted_alt", "A<img alt='cat'>B", "AB"),
    ("upper_case_alt", 'A<img ALT="cat">B', "AB"),
    ("alt_inside_another_name", 'A<img data-alt="cat">B', "AB"),
    ("upper_case_image", 'A<IMG alt="cat">B', "A B"),
)

# The most digits of a decimal string that the interpreter converts to an integer; 0 where it has
# no such limit or the limit is off.
INTEGER_DIGIT_LIMIT = getattr(sys, "get_int_max_str_digits", lambda: 0)()
NO_DIGIT_LIMIT_REASON = "this interpreter converts decimal strings of any length to integers"


def identifiers(cases: tuple[tuple[str, ...], ...]) -> list[str]:
    """Return the name of each case, for the test identifiers."""
    return [case[0] for case in cases]


@pytest.mark.parametrize(("name", "text", "expected"), KNOWN_LIMIT, ids=identifiers(KNOWN_LIMIT))
def test_the_known_limit_of_the_rule(name: str, text: str, expected: str) -> None:
    """Plain text in the form of an accepted tag goes; markup in an unaccepted form stays."""
    assert strip_html(text) == expected


@pytest.mark.parametrize(
    ("name", "text", "expected"), FOREIGN_ELEMENT_SCAN, ids=identifiers(FOREIGN_ELEMENT_SCAN)
)
def test_the_svg_and_math_step_cleans_only_the_spans_its_scan_selects(
    name: str, text: str, expected: str
) -> None:
    """Complete nested spans are cleaned even when an outer span is unclosed or reset."""
    assert strip_html(text) == expected


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    DECLARATIONS_WITH_TAGS_INSIDE,
    ids=identifiers(DECLARATIONS_WITH_TAGS_INSIDE),
)
def test_a_declaration_keeps_its_delimiters_but_not_the_tags_inside_it(
    name: str, text: str, expected: str
) -> None:
    """The delimiters are no tag, so they stay; tags between them go as anywhere else."""
    assert strip_html(text) == expected


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    FORMATTING_RULE_CONDITIONS,
    ids=identifiers(FORMATTING_RULE_CONDITIONS),
)
def test_the_formatting_rules_read_lower_case_tags_and_a_double_quoted_alt(
    name: str, text: str, expected: str
) -> None:
    """A space for ``<BR>`` and ``<IMG>``; nothing for ``<img>`` without alt="..." in lower case."""
    assert strip_html(text) == expected


@pytest.mark.skipif(INTEGER_DIGIT_LIMIT == 0, reason=NO_DIGIT_LIMIT_REASON)
def test_a_decimal_reference_longer_than_the_integer_digit_limit_raises_value_error() -> None:
    """``html.unescape`` refuses the number, and the helper lets its error through."""
    at_the_limit = "&#" + "9" * INTEGER_DIGIT_LIMIT + ";"
    assert strip_html(at_the_limit) == "\ufffd"
    too_long = "&#" + "9" * (INTEGER_DIGIT_LIMIT + 1) + ";"
    with pytest.raises(ValueError, match="integer string conversion"):
        strip_html("a " + too_long + " b")
    assert strip_html("a<!--" + too_long + "-->b") == "a b"
    # A reference kept as the alt text of an image reaches the decoding step.
    assert strip_html('<img alt="&amp;">') == "&"
    with pytest.raises(ValueError, match="integer string conversion"):
        strip_html('<img alt="' + too_long + '">')


def test_cleaning_the_output_of_a_limit_case_again_changes_nothing() -> None:
    """Each output above is plain text to a second cleaning, so it comes back as it is."""
    groups = KNOWN_LIMIT + FOREIGN_ELEMENT_SCAN + DECLARATIONS_WITH_TAGS_INSIDE
    groups += FORMATTING_RULE_CONDITIONS
    for _, text, expected in groups:
        assert strip_html(expected) == expected, text
