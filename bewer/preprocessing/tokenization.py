from typing import TYPE_CHECKING
from typing import Union
from dataclasses import dataclass
from functools import cached_property

import regex as re

from bewer.core.text import TokenList
from bewer.core.token import Token

if TYPE_CHECKING:
    from bewer.core.text import Text


def strip_verbose_pattern(verbose_pattern: str) -> str:
    # Remove comments (lines starting with # or inline)
    pattern = re.sub(r"#.*", "", verbose_pattern)
    # Remove all whitespace (outside character classes)
    pattern = re.sub(r"\s+", "", pattern)
    return pattern


def contains_whitespace(s):
    return any(c.isspace() for c in s)


@dataclass
class StandardTokenPattern:
    """Standard token patterns for text tokenization."""

    # URL pattern (must come before SYMBOL).
    # Matches HTTP/HTTPS or www-prefixed URLs.
    URL = r"""
        (
            https?://[^\s]+                     # Protocol-prefixed URL
            |
            www\.[^\s]+                         # www-prefixed URL
        )
        [^\p{P}\s]                              # Not ending in punctuation or whitespace
    """

    # Email pattern (must come before SYMBOL).
    # Matches standard email formats.
    EMAIL = r"""
        [\p{L}\p{N}._%+-]+                      # Local part
        @
        [\p{L}\p{N}.-]+                         # Domain name
        \.[a-zA-Z]{2,}                          # TLD with at least two letters
    """

    # Punctuation and interrobang pattern.
    # Matches sequences like ?!, !?, and standalone punctuation marks.
    # The \5 backreference (if used in a larger pattern) should be manually updated if group count changes.
    PUNCTUATION = r"""
        (
            \?+!+                               # One or more '?' followed by one or more '!'
            |
            !+\?+                               # One or more '!' followed by one or more '?'
            |
            [.,!?;:–]                           # Common punctuation characters
        )
        \5*                                     # Match same punctuation zero or more times (NOTE: Keep group ref updated)
        (?=\s|$)                                # Lookahead for space or end of string
    """

    # Delimiters like brackets and quotes
    DELIMITER = r"""
        [}{\]\[)(><"]                           # Braces, brackets, parens, angle brackets, quotes
    """

    # Numerical pattern.
    # Matches integers and floats with optional decimal points or commas.
    # Will also capture other numerical formats separated by dots (e.g., IP addresses and dates).
    NUMERICAL = r"""
        (?<=
            [\bx/-]?                           # Lookbehind for optional word boundary or x-/ (e.g., in equations or dates)      
        )
        '?                                      # Optional apostrophe (e.g., in years)
        \d+                                     # One or more digits
        ([.,]\d+)*                              # Optional decimal or comma-separated groups
        (?= 
            [x/-]((?&numerical)|\b|$)           # Option #1: Can be followed x-/ and another number or word boundary
            |
            \b([^']|$)                          # Option #2: Can be followed by a word boundary (unless it's adjacent to an apostrophe)
        )
    """

    # Word pattern.
    # Matches words with letters, numbers, and special characters like '@'.
    # The pattern allows for apostrophes, dashes, and dots within words (for abbreviations).
    WORD = r"""
        [\p{L}\p{N}@]+                          # One or more letters, numbers, or '@'
        (?:
            (?:[-'.@][\p{L}\p{N}]+)             # a quote, dash, dot or @ followed by letters/numbers
            |
            (?:\.(?=(\s+\p{Ll}|,)))             # a dot followed by lookahead: space + lowercase letter or comma
        )*                                      # zero or more of the above groups
        (-(?=[\s/]))?                           # optional dash only if followed by space or slash
    """

    # Generic symbol pattern excluding standard punctuation and delimiter chars
    SYMBOL = r"""
        (?![.,!?;:–}{\]\[)(><"@])               # Exclude these characters
        [\p{P}\p{S}]                            # Match any punctuation or symbol
    """

    @staticmethod
    def minmul() -> str:
        # Special case: Capture x and - as mathematical operators (x and -) or date separators (-) when used alongside numbers.
        lnum = StandardTokenPattern.NUMERICAL + r"\s*"  # Left-side numerical.
        rnum = r"\s*" + StandardTokenPattern.NUMERICAL  # Right-side numerical.
        lwb = rf"(\b|^|{lnum})"  # Left-side word boundary.
        rwb = rf"(\b|$|{rnum})"  # Right-side word boundary.
        minmul = rf"(?<={lwb})[-x](?={rnum})|(?<={lnum})[-x](?={rwb})"
        return minmul

    @staticmethod
    def get_pattern() -> str:
        """Combine all token patterns into a single regex pattern.

        Returns:
            str: The combined regex pattern for tokenization.
        """
        url_group = rf"(?P<word_url>{StandardTokenPattern.URL})"
        email_group = rf"(?P<word_email>{StandardTokenPattern.EMAIL})"
        punctuation_group = rf"(?P<punctuation>{StandardTokenPattern.PUNCTUATION})"
        delimiter_group = rf"(?P<delimiter>{StandardTokenPattern.DELIMITER})"
        minmul_group = rf"(?P<symbol_minmul>{StandardTokenPattern.minmul()})"
        numerical_group = rf"(?P<numerical>{StandardTokenPattern.NUMERICAL})"
        word_group = rf"(?P<word>{StandardTokenPattern.WORD})"
        symbol_group = rf"(?P<symbol>{StandardTokenPattern.SYMBOL})"

        return r"|".join(
            [
                url_group,
                email_group,
                punctuation_group,
                delimiter_group,
                minmul_group,
                numerical_group,
                word_group,
                symbol_group,
            ]
        )


class Tokenizer(object):

    def __init__(
        self,
        pattern: str | None = None,
    ):
        """Initialize a tokenizer with a regex pattern.

        Args:
            pattern (str): The regex pattern to match tokens.
        """
        self._raw_pattern = pattern or StandardTokenPattern.get_pattern()
        self._compiled_base_pattern = re.compile(self._raw_pattern, re.VERBOSE)
        self._groups = list(set([k.split("_")[0] for k in self._compiled_base_pattern.groupindex.keys()]))
        self._nonverbose_pattern = strip_verbose_pattern(self._compiled_base_pattern.pattern)
        self._preview_pattern = (
            self._nonverbose_pattern if len(self._nonverbose_pattern) < 50 else f"{self._nonverbose_pattern[:47]}..."
        )
        self.vocab_items = []

    def add_vocab_items(self, tokens_and_phrases: list[str]) -> None:
        """Add multi-word expressions to the tokenizer."""
        for token_or_phrase in tokens_and_phrases:
            self.vocab_items.append(token_or_phrase.lower())

    @cached_property
    def _compiled_pattern(self) -> str:
        """Get the phrase pattern for multi-word expressions."""
        if not self.vocab_items:
            return self._compiled_base_pattern
        vocab_items = sorted(self.vocab_items, key=len, reverse=True)
        vocab_items = [re.escape(token_or_phrase) for token_or_phrase in vocab_items]
        vocab_items = r"|".join(vocab_items)
        vocab_pattern = rf"(?P<word_vocab>(?i:(?<=(\b|-|\s|^))(?:{vocab_items})(?=(\b|-|\s|$))))"
        return re.compile(rf"{vocab_pattern}|{self._compiled_base_pattern.pattern}", re.VERBOSE)

    def __call__(self, text: "Text") -> list[Token]:
        """Tokenize the input text using the regex pattern.

        Args:
            text (Text): The input text to tokenize.

        Returns:
            list[Token]: A list of Token objects.
        """

        def from_match(match_tuple: tuple[int, re.Match]) -> Token:
            """Create a Token object from a regex match object."""
            token = Token.from_match(match_tuple[1], index=match_tuple[0], _src_text=text)
            return token

        return TokenList(map(from_match, enumerate(self._compiled_pattern.finditer(text.raw))))

    def __repr__(self):
        """Return a string representation of the Tokenizer object."""
        return f'Tokenizer(r"{self._preview_pattern}", groups={self._groups})'
