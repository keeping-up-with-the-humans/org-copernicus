"""The names and patterns that ``strip_html`` reads.

The three lists are those of the cleaning function of the paper's analysis code. The
element names are the HTML standard's index of elements, svg and math included, followed by the
obsolete elements of its list of non-conforming features: 115 and 29 names, taken over without a
date and not checked against a later state of the standard. Outside a complete svg or math
element, a tag of an element that is missing here stays text unless its name has a hyphen or a
colon. The reserved names are the eight hyphenated names that the standard rules out as names of
custom elements. The third list holds the 52 attribute names that the analysis code's function
accepts without a value, among them ``alt`` and the four vendor spellings ``mozallowfullscreen``,
``msallowfullscreen``, ``webkitallowfullscreen`` and ``webkit-playsinline``. The patterns accept
the forms that their comments describe; they do not check that a text follows the syntax of the
standard.
"""

from __future__ import annotations

import re

__all__ = [
    "ATTRIBUTE",
    "ATTRIBUTES",
    "BLANK_LINES",
    "BLOCK_QUOTE_END",
    "BLOCK_QUOTE_START",
    "COMMENT_CLOSER",
    "COMMENT_OPENER",
    "DIVISION_END",
    "DIVISION_START",
    "ELEMENT_NAMES",
    "FOREIGN_ELEMENT_TAG",
    "FOREIGN_TAG",
    "HEADING_END",
    "HEADING_START",
    "IMAGE_TAG",
    "LINE_BREAK",
    "LINE_START_SPACE",
    "LINK",
    "LIST_ITEM_END",
    "LIST_ITEM_START",
    "LIST_TAG",
    "MARKUP",
    "PARAGRAPH_END",
    "PARAGRAPH_START",
    "RESERVED_HYPHENATED_NAMES",
    "SPACE",
    "SPACES",
    "VALUELESS_ATTRIBUTE_NAMES",
]

# ASCII white space, which separates a tag name and its attributes in the HTML standard.
SPACE = r"[\t\n\f\r ]"

# Element names: the HTML standard's index of elements, then the obsolete elements of its list of
# non-conforming features, as the paper's analysis code lists them. Tag names are compared
# without regard to case by Python's Unicode rules, which match more than the standard's ASCII
# rule: "ſ" is read as "s" and "İ" as "i".
ELEMENT_NAMES = frozenset(
    """
    a abbr address area article aside audio b base bdi bdo blockquote body br button canvas
    caption cite code col colgroup data datalist dd del details dfn dialog div dl dt em embed
    fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hgroup hr html i iframe
    img input ins kbd label legend li link main map mark math menu meta meter nav noscript
    object ol optgroup option output p picture pre progress q rp rt ruby s samp script search
    section select selectedcontent slot small source span strong style sub summary sup svg table
    tbody td template textarea tfoot th thead time title tr track u ul var video wbr

    acronym applet basefont bgsound big blink center dir font frame frameset isindex keygen
    listing marquee menuitem multicol nextid nobr noembed noframes param plaintext rb rtc spacer
    strike tt xmp
    """.split()
)

# Hyphenated names that the standard rules out as names of custom elements: they name SVG and
# MathML elements.
RESERVED_HYPHENATED_NAMES = frozenset(
    """
    annotation-xml color-profile font-face font-face-format font-face-name font-face-src
    font-face-uri missing-glyph
    """.split()
)

# The 52 attribute names that the paper's analysis code accepts without "=" and a value, among
# them "alt" and four vendor spellings of allowfullscreen and playsinline in embed codes. Any other
# word after a tag name needs "=" and a value to count as an attribute; a word alone must not, or
# "<i think so>" and "a <b or c> d" would be removed as tags.
VALUELESS_ATTRIBUTE_NAMES = frozenset(
    """
    allowfullscreen alpha async autofocus autoplay checked controls default defer disabled
    formnovalidate inert ismap itemscope loop multiple muted nomodule novalidate open playsinline
    readonly required reversed selected shadowrootclonable shadowrootcustomelementregistry
    shadowrootdelegatesfocus shadowrootserializable

    alt autocorrect contenteditable crossorigin download hidden popover spellcheck translate
    writingsuggestions

    compact declare nohref noresize noshade nowrap seamless truespeed typemustmatch

    mozallowfullscreen msallowfullscreen webkitallowfullscreen webkit-playsinline
    """.split()
)


def _alternation(names: frozenset[str]) -> str:
    """Return a regular-expression alternation of literal names, the longest first.

    Args:
        names: The names to match literally.

    Returns:
        The escaped names joined by ``|``.
    """
    return "|".join(re.escape(name) for name in sorted(names, key=lambda name: (-len(name), name)))


# A character of an attribute name: anything but white space, quotes, ">", "/", "=", "<" and
# control characters.
_ATTRIBUTE_NAME_CHARACTER = r"""[^\t\n\f\r "'>/=<\x00-\x1f\x7f]"""
_ATTRIBUTE_NAME = _ATTRIBUTE_NAME_CHARACTER + "+"
# A value in double quotes, in single quotes, or unquoted without white space, quotes, "=", "<",
# ">" or "`".
_ATTRIBUTE_VALUE = r"""(?:"[^"]*"|'[^']*'|[^\t\n\f\r "'=<>`]+)"""
_VALUED_ATTRIBUTE = rf"{_ATTRIBUTE_NAME}{SPACE}*={SPACE}*{_ATTRIBUTE_VALUE}"
_VALUELESS_ATTRIBUTE = (
    rf"(?i:{_alternation(VALUELESS_ATTRIBUTE_NAMES)})(?!{_ATTRIBUTE_NAME_CHARACTER})"
)
# The attributes of a start tag, each after white space, then the white space before the closing
# "/" or ">".
ATTRIBUTES = rf"(?:{SPACE}+(?:{_VALUED_ATTRIBUTE}|{_VALUELESS_ATTRIBUTE}))*{SPACE}*"
# Inside a selected svg or math span an attribute is a name with "=" and a value in the forms
# above, or a name alone.
_FOREIGN_ATTRIBUTES = rf"(?:{SPACE}+(?:{_VALUED_ATTRIBUTE}|{_ATTRIBUTE_NAME}))*{SPACE}*"

# A tag name: a name of the list of element names; a name in the form of a custom element name
# (an ASCII letter, any ASCII letters or digits, a hyphen, then any ASCII letters, digits or
# hyphens), other than the reserved names; or a namespaced name, two words of ASCII letters,
# digits or hyphens that begin with a letter, joined by a colon, such as "o:p". The name must
# end where white space, "/" or ">" follows, so that "<pine>" is no "<p>" and "<insect>" no
# "<ins>".
_TAG_NAME = (
    rf"(?:(?i:{_alternation(ELEMENT_NAMES)})"
    rf"|(?!(?i:{_alternation(RESERVED_HYPHENATED_NAMES)})[\t\n\f\r />])"
    r"[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9-]*"
    r"|[A-Za-z][A-Za-z0-9-]*:[A-Za-z][A-Za-z0-9-]*)(?=[\t\n\f\r />])"
)

# The rules for single elements read the lower-case name followed by attributes, "/" or ">",
# never a longer word that begins with the name ("<lipstick>" is no "<li>", "<ultimate>" no
# "<ul>").
LINE_BREAK = re.compile(r"<br\s*/?>")
PARAGRAPH_END = re.compile(r"</p>")
PARAGRAPH_START = re.compile(rf"<p{ATTRIBUTES}/?>")
DIVISION_END = re.compile(r"</div>")
DIVISION_START = re.compile(rf"<div{ATTRIBUTES}/?>")
BLOCK_QUOTE_END = re.compile(r"</blockquote>")
BLOCK_QUOTE_START = re.compile(rf"<blockquote{ATTRIBUTES}/?>")
LIST_ITEM_START = re.compile(rf"<li{ATTRIBUTES}/?>")
LIST_ITEM_END = re.compile(r"</li>")
LIST_TAG = re.compile(rf"<[ou]l{ATTRIBUTES}/?>|</[ou]l{SPACE}*>")
HEADING_START = re.compile(rf"<h[1-6]{ATTRIBUTES}/?>")
HEADING_END = re.compile(r"</h[1-6]>")
IMAGE_TAG = re.compile(rf"<img(?P<attributes>{ATTRIBUTES})/?>")
ATTRIBUTE = re.compile(
    rf"(?P<name>{_ATTRIBUTE_NAME})(?:{SPACE}*={SPACE}*(?P<value>{_ATTRIBUTE_VALUE}))?"
)
LINK = re.compile(rf"<a{ATTRIBUTES}/?>(?P<text>[^<]*)</a>")

# The tags that decide where an svg or math element starts and ends, without regard to case: a
# start tag; a "<svg" or "<math" followed by white space, "/" or ">" that begins no such start
# tag (a self-closing one, or one with an attribute outside the forms above), which clears the
# elements open at that place; an end tag. The "<" stands before the alternatives, outside every
# group, so that a search skips to the next "<" at once; with the "<" inside a group, the search
# tries the pattern at every character.
FOREIGN_ELEMENT_TAG = re.compile(
    rf"<(?:(?P<start_name>svg|math){ATTRIBUTES}>"
    r"|(?P<other_start_name>svg|math)[\t\n\f\r />]"
    rf"|/(?P<end_name>svg|math){SPACE}*>)",
    re.IGNORECASE,
)
# A tag inside a selected svg or math span: an end tag or a start tag whose name begins with an
# ASCII letter and runs to white space, "/", "<" or ">", with attributes in the form above.
FOREIGN_TAG = re.compile(
    rf"</[A-Za-z][^\t\n\f\r /<>]*{SPACE}*>|<[A-Za-z][^\t\n\f\r /<>]*{_FOREIGN_ATTRIBUTES}/?>"
)

# A tag that the rules above leave: an end tag or a start tag. Never match from any "<" to the
# next ">": in plain text that deletes everything between an emoticon such as "<3" and a ">" that
# may come thousands of characters later.
MARKUP = re.compile(rf"</{_TAG_NAME}{SPACE}*>|<{_TAG_NAME}{ATTRIBUTES}/?>")

# A comment runs from its opener to the first closer after the opener.
COMMENT_OPENER = "<!--"
COMMENT_CLOSER = "-->"

BLANK_LINES = re.compile(r"\n{3,}")
SPACES = re.compile(r" {2,}")
LINE_START_SPACE = re.compile(r"^\s+", re.MULTILINE)
