from typing import Union
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from functools import cached_property
from itertools import chain

from bewer.core.token import TokenList
from bewer.style.utils import highlight_tokens
from bewer.core.config_vars import TEXT_NORMALIZER
from bewer.core.config_vars import TOKENIZER

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.core.dataset import Dataset


class Text:
    """BeWER text representation.

    Attributes:
        text (str): The original text.
        tokens (list[Token]): A list of Token objects.
    """

    def __init__(
        self,
        raw: str,
        _src_example: Optional["Example"] = None,
        _is_hyp: Optional[bool] = None,
    ):
        """Initialize the Text object.

        Args:
            raw (str): The original text.
            _src (Example): The source Example object.
            _is_hyp (bool): Whether the text is a hypothesis or not.
        """
        self._raw = raw
        self._src_example = _src_example
        self._src_dataset = _src_example._src_dataset if _src_example else None
        self._is_hyp = _is_hyp

    @cached_property
    def raw(self) -> str:
        """Get the raw text.

        Returns:
            str: The raw text.
        """
        if (
            self._is_hyp
            and self._src_dataset is not None
            and isinstance(self._src_dataset._inference_function, Callable)
        ):
            raw, self._src_example._inference_metrics = self._src_dataset._inference_function(self._raw)
            return raw
        return self._raw

    @cached_property
    def tokens(self) -> TokenList:
        """Get the list of Token objects.

        Returns:
            TokenList: The list of Token objects.
        """
        tokenizer = TOKENIZER.get()
        if tokenizer is None:
            raise ValueError("Tokenizer is not set. Please set a tokenizer.")
        return tokenizer(self)

    @cached_property
    def normalized(self) -> str:
        """Get the normalized text.

        Returns:
            str: The normalized text.
        """
        normalized = ""
        prev_end = 0
        for token in self.tokens["!punctuation"]:
            if token.start > prev_end:
                normalized += f" {token.normalized}"
            else:
                normalized += token.normalized
            prev_end = token.end
        return normalized.strip()

    def highlighted(self, tokens: str | TokenList | None = None) -> str:
        """Get tokens (or specific type) highlighted in a zebra striping pattern.

        Args:
            tokens (str | TokenList | None): The tokens to highlight. If None, all tokens are highlighted.

        Returns:
            str: The highlighted text.
        """
        text = self.raw
        if tokens is None:
            tokens = self.tokens
        elif isinstance(tokens, str):
            tokens = self.tokens[tokens]
        elif not isinstance(tokens, TokenList):
            raise ValueError("tokens must be a string, TokenList, or None")

        tokens = self.tokens if type is None else self.tokens[type]
        text = highlight_tokens(text, tokens)
        return text

    def __hash__(self):
        return hash((self._is_hyp, self.raw))

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'Text("{text}")'


class TextList(list[Text]):
    """A list of Text or TokenList objects."""

    @cached_property
    def raw(self) -> list[str]:
        """Get the raw texts as a regular Python list.

        Returns:
            list[str]: The raw text.
        """
        return [text.raw for text in self]

    @cached_property
    def normalized(self) -> list[str]:
        """Get the normalized texts as a regular Python list.

        Returns:
            list[str]: The normalized texts.
        """
        return [text.normalized for text in self]

    @cached_property
    def tokens(self) -> "TextTokenList":
        """Get the tokens as a TextTokenList object.

        Returns:
            TokenList: The tokens.
        """
        return TextTokenList([text.tokens for text in self])

    def __getitem__(self, index: int) -> Union[Text, "TextList"]:
        """Get an item by index.

        Args:
            index (int): The index of the item.

        Returns:
            Token | TextList: The item at the given index.
        """
        if isinstance(index, slice):
            return TextList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "TextList") -> "TextList":
        """Concatenate two TextList objects.

        Args:
            other (TextList): The other TextList object.

        Returns:
            TextList: The concatenated TextList object.
        """
        return TextList(super().__add__(other))

    def __repr__(self):
        texts = self[:60]
        texts_str = ",\n ".join([repr(text) for text in texts])
        if len(self) > 60:
            texts_str += ",\n ..."
        return f"TextList([\n {texts_str}]\n)"


class TextTokenList(list[TokenList]):
    """A list of Text or TokenList objects."""

    @cached_property
    def raw(self) -> list[list[str]]:
        """Get the raw tokens as a regular Python list.

        Returns:
            list[str]: The raw tokens.
        """
        return [tokens.raw for tokens in self]

    @cached_property
    def normalized(self) -> list[str]:
        """Get the normalized tokens as a regular Python list.

        Returns:
            list[str]: The normalized tokens.
        """
        return [tokens.normalized for tokens in self]

    @cached_property
    def flat(self) -> TokenList:
        """Flatten the TextTokenList into a TokenList.

        Returns:
            TokenList: The flattened TokenList.
        """
        return TokenList(chain(*self))

    def __getitem__(self, index: int) -> Union[TokenList, "TextTokenList"]:
        """Get an item by index.

        Args:
            index (int): The index of the item.

        Returns:
            Token | TextList: The item at the given index.
        """
        if isinstance(index, (str, tuple, set, Callable)):
            return TokenList([tokens[index] for tokens in self])
        if isinstance(index, slice):
            return TextTokenList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "TextList") -> "TextList":
        """Concatenate two TextTokenList objects.

        Args:
            other (TextList): The other TextList object.

        Returns:
            TextList: The concatenated TextList object.
        """
        return TextList(super().__add__(other))

    def __repr__(self):
        text_tokens = self[:60]
        text_tokens_str = ",\n ".join([tokens._sub_repr() for tokens in text_tokens])
        if len(self) > 60:
            text_tokens_str += ",\n ..."
        return f"TextTokenList([\n {text_tokens_str}]\n)"
