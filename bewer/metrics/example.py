from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import Optional
from functools import cached_property

from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from bewer.core.example import Example


class ExampleMetric(ABC):

    def __init__(self, _src_example: "Example", key: Optional[Hashable] = None):
        """Initialize the Metric object.

        Args:
            src (Example): The Example to compute the metric for.
        """
        self._src_example = _src_example
        self.key = key
        self._members = {}

    @property
    @abstractmethod
    def long_name(self) -> str:
        """Get the long/full name of the metric."""
        pass

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Get the short name (e.g., an abbreviation) of the metric."""
        pass

    @property
    @abstractmethod
    def attr_name(self) -> str:
        """Get the attribute name of the metric."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of the metric."""
        pass

    @property
    @abstractmethod
    def value(self) -> float:
        """Compute the metric value."""
        pass

    @property
    @abstractmethod
    def rvalue(self) -> float:
        """Get the rounded metric value."""
        pass

    @property
    @abstractmethod
    def fvalue(self) -> str:
        """Get the formatted metric value."""
        pass

    def is_valid_key(self, key: Hashable) -> bool:
        """Check if the filter is valid."""
        raise NotImplementedError(f"Metric '{self.short_name}' does not support filtering.")

    def __getitem__(self, key: Hashable) -> "ExampleMetric":
        """Get a metric by a hashable key."""
        if key in self._members:
            return self._members[key]
        elif self.is_valid_key(key):
            self._members[key] = self.__class__(self._src_example, key)
            return self._members[key]
        raise AttributeError(f"'{key}' is not a valid key for this metric.")

    def __repr__(self):
        if self.key is None:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({self.key})"


class WER(ExampleMetric):
    """Word Error Rate (WER) metric representation."""

    short_name = "WER"
    long_name = "Word Error Rate"
    attr_name = "wer"
    description = (
        "Word Error Rate (WER) is computed as the token-level (i.e., word-level) edit distance between the reference "
        "and hypothesis text, divided by the total number of tokens in the reference text."
    )

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits."""
        return self._src_example.levenshtein.edits

    @cached_property
    def num_ref_tokens(self) -> int:
        """Get the length of the reference text."""
        return len(self._src_example.ref.tokens["!punctuation"])

    @cached_property
    def value(self) -> float:
        if self.num_ref_tokens == 0:
            return 0.0
        return self.num_edits / self.num_ref_tokens

    @cached_property
    def rvalue(self) -> float:
        return round(self.value, 4)

    @cached_property
    def fvalue(self) -> str:
        return f"{self.value * 100:.2f}%"


class CER(ExampleMetric):
    """Character Error Rate (WER) metric representation."""

    short_name = "CER"
    long_name = "Character Error Rate"
    attr_name = "cer"
    description = (
        "Character Error Rate (CER) is computed as the character-level edit distance between the normalized reference "
        "and hypothesis text, divided by the total number of characters in the reference text."
    )

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits."""
        return Levenshtein.distance(self._src_example.hyp.normalized, self._src_example.ref.normalized)

    @cached_property
    def num_ref_chars(self) -> int:
        """Get the length of the reference text."""
        return len(self._src_example.ref.normalized)

    @cached_property
    def value(self) -> float:
        if self.num_ref_chars == 0:
            return 0.0
        return self.num_edits / self.num_ref_chars

    @cached_property
    def rvalue(self) -> float:
        return round(self.value, 4)

    @cached_property
    def fvalue(self) -> str:
        return f"{self.value * 100:.2f}%"
