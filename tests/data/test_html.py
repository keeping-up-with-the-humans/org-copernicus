"""strip_html removes comments and the spans that have the form of a tag it accepts.

A ``<`` goes only as part of such a span. The strings are made up. Each group covers one rule of
the helper, and the expected outputs are literals. The limits of the rule are pinned in
``test_html_limits.py`` and its running time in ``test_html_running_time.py``.
"""

from __future__ import annotations

import pytest

from copernicus.data import strip_html

# Plain text in which no "<" begins a span that the rule accepts: each text comes back unchanged.
TEXT_WITHOUT_TAGS: tuple[tuple[str, str], ...] = (
    ("emoticon_then_greater_than", "I <3 this film. It is > all the rest."),
    ("emoticon_then_a_later_greater_than", "I love this <3 so much, he said > nope. Next stays."),
    ("comparisons", "a < b and c > d"),
    ("repeated_less_than", "<<< look"),
    ("empty_pair_with_a_space", "< >"),
    ("token", "<|token|>"),
    ("two_tokens", "<|start|> hello <|end|>"),
    ("emoticon_in_brackets", "we met <3 times> last year"),
    ("arrows", "turn left <- here -> then right"),
    ("less_or_equal", "if a <= b and b >= c"),
    ("number", "the box <42> is full"),
    ("placeholder", "Dear <name of the guest>, welcome."),
    ("stage_direction", "<long pause> Fine, go on."),
    ("pseudo_empty_tags", "<smile/> and <frown />"),
    ("exclamations", "<!!!> wow"),
    ("slash_only", "</> closed"),
    ("dots", "<..> dots"),
    ("bracketed_link", "<https://example.org/page>"),
    ("element_name_with_words", "x <b or c> y"),
    ("element_name_with_a_sentence", "<i think so> she said"),
    ("double_arrow", "a <--> b"),
    ("letters_outside_ascii", "<ébauche> d'idée 🙂"),
)

# A "<" that no ">" follows, inside a text or at its end, and the empty pair "<>".
LONE_LESS_THAN_SIGNS: tuple[tuple[str, str], ...] = (
    ("emoticon_at_the_end", "she sent <3"),
    ("only_less_than_signs", "<<<"),
    ("arrow_at_the_end", "go back <-"),
    ("less_or_equal_alone", "x <= y"),
    ("less_than_at_the_end", "the last char is <"),
    ("empty_pair_then_greater_than", "scores <> 4 and > 2"),
)

# Words after "<" that only begin like a tag that a rule of its own handles, and character
# references that are not closed by ";" or that name nothing: each text comes back unchanged.
TEXT_THAT_BEGINS_LIKE_A_HANDLED_TAG: tuple[tuple[str, str], ...] = (
    ("paragraph", "a <pine tree> b"),
    ("division", "the <divine light> shines"),
    ("block_quote", "a <blockquotes galore> here"),
    ("list_item", "my <lipstick> case"),
    ("unordered_list", "the <ultimate goal> is near"),
    ("ordered_list", "back in the <olden days>"),
    ("heading", "a <h2o bottle> please"),
    ("image", "the <imgsrc folder> moved"),
    ("link", "<about us> page"),
    ("legacy_reference_in_a_word", "Kind&regards, reader"),
    ("legacy_references_in_a_link", "https://example.org/?a=1&copy=2&para=3"),
    ("undefined_reference", "the &notaname; stays"),
)

# Real markup and the text each rule makes of it.
REAL_MARKUP: tuple[tuple[str, str, str], ...] = (
    ("line_breaks", "one<br>two<br/>three<br />four", "one\ntwo\nthree\nfour"),
    ("paragraphs", '<p>First.</p><p class="note">Second.</p>', "First.\nSecond."),
    ("divisions", '<div id="a">Left</div><div>Right</div>', "Left\nRight"),
    ("block_quote", "<blockquote>Said before</blockquote>Answer", "> Said before\nAnswer"),
    (
        "lists",
        '<ul><li>red</li><li>blue</li></ul><ol start="3"><li>green</li></ol>',
        "- red\n- blue\n- green",
    ),
    ("headings", '<h1>Title</h1>Body<h3 id="x">Part</h3>', "Title\nBody\nPart"),
    ("image_with_alt", 'See <img src="dog.png" alt="a small dog"> there', "See a small dog there"),
    ("image_without_alt", 'See <img src="dog.png"> there', "See there"),
    ("link", 'Read <a href="https://example.org" rel="nofollow">the notes</a>.', "Read the notes."),
    (
        "inline_tags",
        '<b>bold</b> and <i>slanted</i> and <span class="x">plain</span>',
        "bold and slanted and plain",
    ),
    ("single_quoted_attribute", "<span class='x'>kept</span>", "kept"),
    ("paragraph_and_comment", "<p>Hello <b>you</b></p><!-- note -->", "Hello you"),
    ("comment", "before<!-- hidden -->after", "before after"),
    ("escaped_markup", "&lt;i&gt;not slanted&lt;/i&gt;", "<i>not slanted</i>"),
    (
        "svg_icon",
        '<svg class="icon" aria-hidden="true"><use href="#star"></use></svg>star.png',
        "star.png",
    ),
    ("custom_element", '<photo-card data-id="7">A caption</photo-card>', "A caption"),
    ("namespaced_element", "Some text<o:p></o:p> more", "Some text more"),
    ("upper_case_tags", "<B>Loud</B> <DIV>box</DIV>", "Loud box"),
    ("valueless_attribute", "<details open><summary>Title</summary>Body</details>", "Title Body"),
    (
        "embed_code",
        '<iframe src="https://video.example/embed/1" width="560" frameborder="0" '
        "webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>Clip",
        "Clip",
    ),
    ("math_element", "<math><mi>y</mi><mo>=</mo><mn>3</mn></math> holds", "y = 3 holds"),
    (
        "nested_foreign_elements",
        '<svg><g><svg><use href="#a"></use>p</svg></g><math><mi>q</mi></math></svg>r',
        "p q r",
    ),
    (
        "quoted_post",
        '<aside class="quote">\n<div class="title">\n<img alt="" src="/face.png"> user:</div>\n'
        "<blockquote>\n<p>Earlier words</p>\n</blockquote>\n</aside>\n<p>My answer &amp; more</p>",
        "user:\n> \nEarlier words\nMy answer & more",
    ),
    ("white_space", "a  b\n\n\n\nc\n   d\t e", "a b\nc\nd\t e"),
    ("greater_than_inside_a_quoted_attribute", '<a title="x>y" href="z">link</a> end', "link end"),
    ("greater_than_inside_a_comment", "a<!-- p > q -->b", "a b"),
    ("doctype_stays_text", "<!DOCTYPE html><p>x</p>", "<!DOCTYPE html>x"),
    ("processing_instruction_stays_text", "<?xml version='1.0'?>y", "<?xml version='1.0'?>y"),
    ("cdata_section_stays_text", "<![CDATA[z]]>w", "<![CDATA[z]]>w"),
)

MIXED_TEXT = (
    "<p>I <3 this <b>tune</b> &amp; x < y and z > w</p>\n"
    '<p>Dear <name here>, see <a href="https://example.org">this</a></p>'
)

# Where a comment ends: at the first "-->" after "<!--", never at the dashes of the opener itself.
# An opener that no "-->" follows is text, and the tags after it are still markup.
COMMENT_BOUNDARIES: tuple[tuple[str, str, str], ...] = (
    ("empty_comment", "a<!---->b", "a b"),
    ("opener_inside_a_comment", "a<!-- <!-- -->b", "a b"),
    ("two_comments", "x<!-- a -->y<!-- b -->z", "x y z"),
    ("comment_then_an_unclosed_opener", "a<!-- x -->b<!-- y", "a b<!-- y"),
    ("unclosed_opener_then_tags", "a<!-- b <i>c</i> d", "a<!-- b c d"),
    ("opener_closed_only_by_its_own_dashes", "a <!--> b", "a <!--> b"),
    ("opener_with_three_dashes", "a <!---> b", "a <!---> b"),
    ("comment_after_a_tag_that_holds_an_opener", '<b title="<!--">x</b><!-- c -->y', "x y"),
)

# A comment and a tag that overlap: the one that starts first takes the other with it, as an HTML
# reader reads them.
OVERLAPPING_COMMENT_AND_TAG: tuple[tuple[str, str, str], ...] = (
    ("comment_opener_inside_a_quoted_attribute", 'x<b title="<!--">y</b>-->z', "x y -->z"),
    ("tag_inside_a_comment_reaching_past_its_end", 'a<!-- <b title=" -->">c', 'a ">c'),
)

# Only character references closed by ";" are decoded, a named one only when the HTML standard
# defines its name.
CHARACTER_REFERENCES: tuple[tuple[str, str, str], ...] = (
    ("named", "&copy; 2026", "© 2026"),
    ("named_without_semicolon", "&copy 2026", "&copy 2026"),
    ("numbers", "&#169; and &#xA9;", "© and ©"),
    ("numbers_without_semicolon", "&#169 and &#xA9 stay", "&#169 and &#xA9 stay"),
    ("escaped_reference", "&amp;amp;", "&amp;"),
    ("number_that_decodes_to_nothing", "&#1;", ""),
    ("name_that_decodes_to_two_characters", "&fjlig;", "fj"),
    (
        "common_references",
        "&amp; &lt; &gt; &quot; &#39; &#x27; &nbsp;x &hellip; &zzzz;",
        "& < > \" ' ' \xa0x … &zzzz;",
    ),
)

# Inputs whose output a second cleaning changes: escaped markup and escaped references are decoded
# once, after the markup is gone, into text that a second cleaning reads again.
ESCAPED = frozenset({"escaped_markup", "escaped_reference"})


def identifiers(cases: tuple[tuple[str, ...], ...]) -> list[str]:
    """Return the name of each case, for the test identifiers."""
    return [case[0] for case in cases]


@pytest.mark.parametrize(("name", "text"), TEXT_WITHOUT_TAGS, ids=identifiers(TEXT_WITHOUT_TAGS))
def test_a_less_than_sign_that_opens_no_tag_stays_with_what_follows(name: str, text: str) -> None:
    """These texts hold no span that the rule accepts, so each comes back unchanged."""
    assert strip_html(text) == text


@pytest.mark.parametrize(
    ("name", "text"), LONE_LESS_THAN_SIGNS, ids=identifiers(LONE_LESS_THAN_SIGNS)
)
def test_a_lone_less_than_sign_or_an_empty_pair_stays(name: str, text: str) -> None:
    """A "<" that no ">" follows stays, at the end of a text as well, and so does "<>"."""
    assert strip_html(text) == text


@pytest.mark.parametrize(
    ("name", "text"),
    TEXT_THAT_BEGINS_LIKE_A_HANDLED_TAG,
    ids=identifiers(TEXT_THAT_BEGINS_LIKE_A_HANDLED_TAG),
)
def test_a_word_that_only_begins_like_a_handled_tag_stays(name: str, text: str) -> None:
    """The rules for p, div, blockquote, li, ul, ol, h1 to h6, img and a read whole names."""
    assert strip_html(text) == text


def test_the_link_rule_needs_a_link_tag() -> None:
    """``<a`` followed by words that are no attributes is no link; only ``</a>`` is markup."""
    assert strip_html("<a big hug>for you</a>") == "<a big hug>for you"


@pytest.mark.parametrize(("name", "text", "expected"), REAL_MARKUP, ids=identifiers(REAL_MARKUP))
def test_each_rule_turns_real_markup_into_text(name: str, text: str, expected: str) -> None:
    """Each rule turns real markup into the expected text."""
    assert strip_html(text) == expected


def test_markup_and_plain_angle_brackets_in_one_text() -> None:
    """The tags go, and the plain "<" and ">" of the same text stay."""
    expected = "I <3 this tune & x < y and z > w\nDear <name here>, see this"
    assert strip_html(MIXED_TEXT) == expected


@pytest.mark.parametrize(
    ("name", "text", "expected"), COMMENT_BOUNDARIES, ids=identifiers(COMMENT_BOUNDARIES)
)
def test_a_comment_ends_at_the_first_closer_after_its_opener(
    name: str, text: str, expected: str
) -> None:
    """A comment runs from "<!--" to the first "-->" after it; an unclosed opener stays text."""
    assert strip_html(text) == expected


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    OVERLAPPING_COMMENT_AND_TAG,
    ids=identifiers(OVERLAPPING_COMMENT_AND_TAG),
)
def test_comments_and_tags_are_read_in_one_pass_from_left_to_right(
    name: str, text: str, expected: str
) -> None:
    """A "<!--" inside a quoted value belongs to its tag; a tag inside a comment to the comment."""
    assert strip_html(text) == expected


@pytest.mark.parametrize(
    ("name", "text", "expected"), CHARACTER_REFERENCES, ids=identifiers(CHARACTER_REFERENCES)
)
def test_only_character_references_closed_by_a_semicolon_are_decoded(
    name: str, text: str, expected: str
) -> None:
    """A reference without ";" and a name that the standard does not define stay as written."""
    assert strip_html(text) == expected


def test_a_second_cleaning_leaves_the_outputs_above_and_removes_a_joined_span() -> None:
    """A second cleaning leaves the outputs above as they are; a joined span shows the exception.

    Every output of the made-up inputs above but the escaped ones stays as it is when it is
    cleaned again. Removing a span can join the text around it into a new tag, which a second
    cleaning removes, so a second cleaning can change an output.
    """
    plain = TEXT_WITHOUT_TAGS + LONE_LESS_THAN_SIGNS + TEXT_THAT_BEGINS_LIKE_A_HANDLED_TAG
    texts = [text for _, text in plain] + [MIXED_TEXT]
    groups = REAL_MARKUP + COMMENT_BOUNDARIES + OVERLAPPING_COMMENT_AND_TAG
    groups += CHARACTER_REFERENCES
    texts += [text for name, text, _ in groups if name not in ESCAPED]
    for text in texts:
        once = strip_html(text)
        assert strip_html(once) == once, text
    joined = strip_html("<b<i>>x</b<i>>")
    assert joined == "<b >x</b >"
    assert strip_html(joined) == "x"


def test_escaped_markup_is_decoded_once_and_read_as_markup_by_a_second_cleaning() -> None:
    """References are decoded after the markup is gone; the decoded ``<b>`` is markup next time."""
    once = strip_html("&lt;b&gt;x&lt;/b&gt;")
    assert once == "<b>x</b>"
    assert strip_html(once) == "x"


@pytest.mark.parametrize("value", [None, 5, b"<p>x</p>"], ids=["none", "number", "bytes"])
def test_a_text_that_is_not_a_string_is_refused(value: object) -> None:
    """A missing text must not pass as an empty one."""
    with pytest.raises(TypeError, match="needs the text as a string, got"):
        strip_html(value)  # type: ignore[arg-type]


def test_an_empty_text_gives_an_empty_text() -> None:
    """An empty string is a text, and its plain text is empty."""
    assert strip_html("") == ""
