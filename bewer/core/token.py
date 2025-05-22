import regex as re
from typing import Union
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from functools import cached_property

from bewer.style.utils import highlight_span
from bewer.core.config_vars import TOKEN_NORMALIZER

if TYPE_CHECKING:
    from bewer.core.text import Text
    from bewer.core.op import Op
    

def _join_tokens(tokens: "TokenList", normalized: bool = True, delimiter: str = " ") -> str:
    """Join a list of tokens into a single string with a specified delimiter.

    Args:
        tokens (list[str]): The list of tokens to join.
        delimiter (str): The delimiter to use for joining.

    Returns:
        str: The joined string.
    """
    joined = ""
    prev_end = 0
    for token in tokens:
        if token.start > prev_end:
            joined += f" {token.normalized}" if normalized else f" {token.raw}"
        else:
            joined += token.normalized  if normalized else token.raw
        prev_end = token.end
    return joined.strip()


class Token:
    """BeWER Token representation.

    Attributes:
        start (int): The starting index of the token in the text.
        end (int): The ending index of the token in the text.
        raw (str): The raw string of the token.
        slice (slice): A slice object representing the token's position in the text.
        normalized (str | None): The normalized string of the token.
    """

    def __init__(
        self,
        raw: str,
        start: int,
        end: int,
        index: Optional[int] = None,
        type: Optional[str] = None,
        _src_text: Optional["Text"] = None,
    ):
        self.raw = raw
        self.start = start
        self.end = end
        self.index = index
        self._src_text = _src_text
        self._src_example = _src_text._src_example if _src_text else None
        self.type = type
        self.slice = slice(self.start, self.end)

    @cached_property
    def normalized(self) -> str:
        """Get the normalized string of the token.

        Returns:
            str: The normalized string.
        """
        normalizer = TOKEN_NORMALIZER.get()
        if normalizer is not None:
            return normalizer(self.raw)
        return None

    @cached_property
    def punctuation(self) -> str:
        """Get the punctuation of the token.

        Returns:
            str: The punctuation string.
        """
        if self.type == "punctuation":
            return None
        if self.index + 1 < len(self._src_text.tokens):
            if self._src_text.tokens[self.index + 1].type == "punctuation":
                return self._src_text.tokens[self.index + 1].raw
        return None

    @cached_property
    def levenshtein(self) -> "Op":
        if self._src_example is None:
            return None
        return self._src_example.levenshtein._token_to_op_index.get(self, None)

    def inctx(self, width: int = 20, highlight: bool = False, add_delim_markers: bool = True) -> str:
        """Get the context of the token in the source text.

        Args:
            width (int): The width of the context to show.
            add_delim_markers (bool): Whether to add delimiters around the context.

        Returns:
            str: The context string.
        """
        if self._src_text is None:
            raise ValueError("Source text is not set. Cannot get context.")
        start = max(0, self.start - width)
        end = min(len(self._src_text.raw), self.end + width)
        ctx_span = self._src_text.raw[start:end]
        if highlight:
            ctx_span = highlight_span(ctx_span, self.start - start, self.end - start, "bold green")
        if add_delim_markers:
            start_marker = "..." if self.start - width > 0 else ""
            end_marker = "..." if self.end + width < len(self._src_text.raw) else ""
            ctx_span = start_marker + ctx_span + end_marker
        return ctx_span

    @classmethod
    def from_match(
        cls,
        match: re.Match,
        index: int,
        add_type: bool = True,
        _src_text: Optional["Text"] = None,
    ) -> "Token":
        """
        Create a Token object from a regex match object.

        Args:
            match (re.Match): The regex match object.

        Returns:
            Token: The created Token object.
        """
        token_type = None
        if add_type:
            token_types = [k for k, v in match.groupdict().items() if v is not None]
            if len(token_types) > 0:
                token_type = token_types[0].split("_")[0]
        return cls(
            # NOTE: Restore phrases. This should be changed if we want to allow underscores in transcriptions.
            raw=match.group().replace("_", " "),
            start=match.start(),
            end=match.end(),
            index=index,
            type=token_type,
            _src_text=_src_text,
        )

    def __hash__(self):
        return hash((self._src_text.__hash__(), self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.start == other.start and self.end == other.end and self.raw == other.raw and self.type == other.type

    def __repr__(self):
        if self.type is None:
            return f'Token("{self.raw}")'
        else:
            return f'Token("{self.raw}", type="{self.type}")'


class TokenList(list[Token]):
    """A list of Token objects."""

    @cached_property
    def raw(self) -> list[str]:
        """Get the raw tokens as a regular Python list.

        Returns:
            list[str]: The raw tokens.
        """
        return [token.raw for token in self]

    @cached_property
    def normalized(self) -> list[str]:
        """Get the normalized tokens as a regular Python list.

        Returns:
            list[str]: The normalized tokens.
        """
        return [token.normalized for token in self]

    def ngrams(
        self,
        n: int,
        normalized: bool = True,
        join_tokens: bool = True,
        ignore_punctuation: bool = False,
    ) -> list[str]:
        """Get n-grams from the token list.

        Args:
            n (int): The size of the n-grams.
            normalized (bool): Whether to use normalized tokens.
            ignore_punctuation (bool): Whether to return n-grams that span punctuation. If False, n-grams that span
                punctuation will be ignored.

        Returns:
            list[str]: The list of n-grams.
        """
        if n < 1:
            raise ValueError("n must be a positive integer")
        if n == 1:
            return self.raw if normalized else self.normalized
        ngrams = []
        tokens = self["!punctuation"] if ignore_punctuation else self
        for i in range(len(tokens) - n + 1):
            ngram = tokens[i : i + n]
            if not ignore_punctuation and any(token.type == "punctuation" for token in ngram):
                continue
            if join_tokens:
                ngram = _join_tokens(ngram, normalized=normalized)
            else:
                ngram = (ngram.normalized if normalized else ngram.raw)
            ngrams.append(ngram)
        return ngrams

    def _sub_repr(self):
        """Get a string representation of the TokenList object.

        Returns:
            str: The string representation.
        """
        tokens = self[:5]
        tokens_str = ",  ".join([repr(token) for token in tokens])
        if len(self) > 5:
            tokens_str += ", ..."
        return f"TokenList([{tokens_str}])"

    def __getitem__(self, index: int) -> Union[Token, "TokenList"]:
        if isinstance(index, slice):
            return TokenList(super().__getitem__(index))
        if isinstance(index, str):
            if index.startswith("!"):
                return TokenList(filter(lambda token: token.type != index[1:], self))
            return TokenList(filter(lambda token: token.type == index, self))
        if isinstance(index, (tuple, set)):
            if len(index) < 1:
                raise ValueError("index must be a non-empty tuple or set")
            if not isinstance(next(map(type, index)), str):
                raise TypeError("tuple-index types must be string")
            return TokenList(filter(lambda token: token.type in index, self))
        if isinstance(index, Callable):
            return TokenList(filter(index, self))
        return super().__getitem__(index)

    def __add__(self, other: "TokenList") -> "TokenList":
        """Concatenate two TokenList objects.

        Args:
            other (TokenList): The other TokenList object.

        Returns:
            TokenList: The concatenated TokenList object.
        """
        return TokenList(super().__add__(other))

    def __repr__(self):
        tokens = self[:60]
        tokens_str = ",\n ".join([repr(token) for token in tokens])
        if len(self) > 60:
            tokens_str += ",\n ..."
        return f"TokenList([\n {tokens_str}]\n)"
