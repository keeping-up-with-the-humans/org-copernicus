"""Turn a text that may hold HTML into plain text, by the rules of ``strip_html``."""

from __future__ import annotations

import logging

from copernicus.data._html_references import decode_character_references
from copernicus.data._html_steps import (
    image_alt_text,
    remove_foreign_elements,
    replace_comments_and_tags,
)
from copernicus.data._html_syntax import (
    BLANK_LINES,
    BLOCK_QUOTE_END,
    BLOCK_QUOTE_START,
    DIVISION_END,
    DIVISION_START,
    HEADING_END,
    HEADING_START,
    IMAGE_TAG,
    LINE_BREAK,
    LINE_START_SPACE,
    LINK,
    LIST_ITEM_END,
    LIST_ITEM_START,
    LIST_TAG,
    MARKUP,
    PARAGRAPH_END,
    PARAGRAPH_START,
    SPACES,
)

logger = logging.getLogger(__name__)

__all__ = ["strip_html"]


def strip_html(text: str) -> str:
    """Turn a text that may hold HTML into plain text.

    A ``<`` is removed only as part of a span that the rules below accept, so a ``<`` that is
    not part of such a span stays, and the text after it is read on by the same rules instead of
    being removed with it. The rules read tags by their form, not by what the writer meant: see
    the known limit below.

    The rules run in this order:

    1. Line breaks and blocks, in lower-case tags: ``<br>`` without attributes (also ``<br/>``,
       ``<br />``) becomes a line break; the start tags of ``p`` and ``div`` are removed and
       their end tags become line breaks; ``<blockquote>`` becomes a line break and ``"> "``,
       ``</blockquote>`` a line break; ``<li>`` becomes a line break and ``"- "``, ``</li>`` is
       removed; the start and end tags of ``ul``, ``ol`` and ``h1`` to ``h6`` become line
       breaks. Each rule reads the whole element name, so ``<lipstick>`` is no list item. A tag
       that these rules do not match, such as ``<BR>``, ``<br class="x">`` or ``<P>``, is left
       to rule 4, which removes it if it accepts its form and leaves a space: ``A<BR>B`` gives
       ``A B``.
    2. Images and links, in lower-case tags as well: an ``img`` tag becomes the value of its
       last ``alt`` attribute that is written in lower case and has its value in double quotes,
       and otherwise nothing, so ``<img alt='cat'>`` and ``<img data-alt="cat">`` give nothing;
       ``<a ...>text</a>`` becomes its text when the text holds no ``<``. An upper-case
       ``<IMG ...>`` is left to rule 4.
    3. ``svg`` and ``math`` elements: a scan from left to right tracks the accepted start tags of
       ``svg`` and ``math``, in either case and with attributes in the form of rule 4 but no
       closing ``/``, and their matching end tags. An ``<svg`` or ``<math`` followed by ASCII
       white space, ``/`` or ``>`` that does not begin an accepted start tag, such as a
       self-closing ``<svg/>`` or ``<math/>`` or a start tag with an attribute that rule 4 does
       not accept, clears the elements that are open at that place, also when it stands inside
       a comment; ``<svg!>`` does not. Each outermost span from an accepted start tag to its
       matching end tag is selected. An outer element that is never completed, or that a reset
       clears, is not selected as a whole, but a complete span nested in it is still selected,
       before or after the reset: ``<svg><svg/><math><text>Hello</text></math></svg>`` gives
       ``Hello``. Within each selected span, its own tags, its comments and the tags in the form
       that this step accepts become spaces: a tag whose name begins with an ASCII letter and
       whose attributes are names, with or without a value in the form of rule 4. The text and
       everything else pass to rule 4, which keeps a tag whose name is not on its list and has
       neither a hyphen nor a colon: ``<svg><svg/><text>Hello</text></svg>`` gives
       ``<text>Hello</text>``, and ``<svg><é>Hello</é></svg>`` gives ``<é>Hello</é>``.
    4. Comments and tags: every other span that has the form of a comment or of a tag that the
       rule accepts becomes a space. A comment runs from ``<!--`` to the first ``-->`` after it;
       an opener that no ``-->`` follows is text. Comments and tags are read in one pass from
       left to right, so a ``<!--`` inside a quoted attribute value belongs to its tag and a tag
       that starts inside a comment belongs to the comment. A tag is ``<``, an optional ``/``, a
       tag name, then, in a start tag, its attributes and an optional ``/``, then ``>``. A tag
       name is a name on the rule's list of element names, compared without regard to case by
       Python's Unicode rules (144 names: the HTML standard's index of elements, svg and math
       included, with its obsolete elements, as the paper's analysis code lists them, without a
       date); a name in the form of a custom element name (an ASCII letter, any ASCII letters or
       digits, a hyphen, then any ASCII letters, digits or hyphens), other than eight reserved
       names such as ``font-face``; or a namespaced name, two words of ASCII letters, digits or
       hyphens that begin with a letter, joined by a colon, such as ``o:p``. Each attribute
       follows white space and is a name with ``=`` and a value (quoted, or unquoted without
       white space, quotes, ``=``, ``<``, ``>`` or a backtick), or a name on the rule's list of
       52 attributes that may stand without a value (``open``, ``checked``, ``allowfullscreen``
       and others, among them the vendor spellings ``mozallowfullscreen``,
       ``msallowfullscreen``, ``webkitallowfullscreen`` and ``webkit-playsinline``). Spans of
       other forms are not removed: ``<3``, ``<<<``, ``< >``, ``a < b and c > d``, ``<-``,
       ``<=``, ``<42>``, ``<name of the guest>``, ``x <b or c> y``, ``<|token|>``,
       ``<https://example.org>`` and a ``<`` at the end of the text. A doctype, a processing
       instruction and a CDATA section are not read as tags, so their delimiters stay, while
       their contents are read by these rules like any text: ``<![CDATA[<b>word</b>]]>`` gives
       ``<![CDATA[ word ]]>``.
    5. Character references closed by ``;`` are decoded: a name that the HTML standard defines
       (the table ``html.entities.html5``, such as ``&amp;`` or ``&hellip;``) or a number
       (``&#39;``, ``&#x27;``). A reference without ``;`` (the ``&copy`` of ``?id=5&copy=2``)
       and an undefined name (``&notaname;``) stay as written.
    6. White space: runs of three or more line breaks become two and runs of spaces one space;
       white space at the start of every line is removed, which removes blank lines too, so
       paragraphs end one line break apart; the ends are stripped.

    The rule has a known limit: plain text between angle brackets is removed when it has the
    form of a tag that rule 4 accepts. That covers three classes: a word that is an element
    name, alone or with a word that may stand as an attribute without a value (``<A>``,
    ``<B>``, ``<i open>``, ``<link>``, ``<time>``; by Python's Unicode rules also ``<ſ>`` and
    ``<İ>``, read as ``s`` and ``i``); a word with a hyphen, which has the form of a custom
    element name (``<well-known>``); and two words joined by a colon, the form of a namespaced
    name (``<Note:important>``). The other way round, real markup with an attribute without a
    value that the rule's list does not hold stays text, such as ``<div data-flag>`` or
    ``<img alt="cat" lazy>``.

    Cleaning the result again can change it. Escaped markup and escaped references
    (``&lt;b&gt;``, ``&amp;amp;``) are decoded once, after the markup is gone, into text that a
    second cleaning reads as markup or decodes again. And removing a span can join the text
    around it into a new one: ``<b<i>>x</b<i>>`` gives ``<b >x</b >``, which a second cleaning
    turns into ``x``.

    Args:
        text: The text, HTML or plain text.

    Returns:
        The plain text; an empty string for an empty text.

    Raises:
        TypeError: If ``text`` is not a string, so that a missing text never passes as an empty
            one.
        ValueError: If a decimal character reference closed by ``;`` reaches rule 5 and holds
            more digits than the interpreter converts to an integer
            (``sys.get_int_max_str_digits()``, 4,300 by default where Python has the limit);
            ``html.unescape`` raises it. A reference that an earlier rule discards with a comment
            or a tag never reaches rule 5; one kept as the ``alt`` text of an image does.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"strip_html needs the text as a string, got {type(text).__name__}: pass the text "
            "itself, and decide before the call what a missing text stands for"
        )
    cleaned = LINE_BREAK.sub("\n", text)
    cleaned = PARAGRAPH_END.sub("\n", cleaned)
    cleaned = PARAGRAPH_START.sub("", cleaned)
    cleaned = DIVISION_END.sub("\n", cleaned)
    cleaned = DIVISION_START.sub("", cleaned)
    cleaned = BLOCK_QUOTE_END.sub("\n", cleaned)
    cleaned = BLOCK_QUOTE_START.sub("\n> ", cleaned)
    cleaned = LIST_ITEM_START.sub("\n- ", cleaned)
    cleaned = LIST_ITEM_END.sub("", cleaned)
    cleaned = LIST_TAG.sub("\n", cleaned)
    cleaned = HEADING_START.sub("\n", cleaned)
    cleaned = HEADING_END.sub("\n", cleaned)
    cleaned = IMAGE_TAG.sub(image_alt_text, cleaned)
    cleaned = LINK.sub(r"\g<text>", cleaned)
    cleaned = remove_foreign_elements(cleaned)
    cleaned = replace_comments_and_tags(cleaned, MARKUP, 0, len(cleaned))
    cleaned = decode_character_references(cleaned)
    cleaned = BLANK_LINES.sub("\n\n", cleaned)
    cleaned = SPACES.sub(" ", cleaned)
    cleaned = LINE_START_SPACE.sub("", cleaned)
    return cleaned.strip()
