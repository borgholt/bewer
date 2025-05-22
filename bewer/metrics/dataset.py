from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import Optional
from functools import cached_property

from rapidfuzz.distance import Levenshtein

from bewer.core.op import OpType

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset


class Metric(ABC):

    def __init__(self, _src_dataset: "Dataset", key: Optional[Hashable] = None):
        """Initialize the Metric object.

        Args:
            src (Dataset): The dataset to compute the metric for.
        """
        self._src_dataset = _src_dataset
        self.key = key
        self._src_dataset._locked = True
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

    def __getitem__(self, key: Hashable) -> "Metric":
        """Get a metric by a hashable key."""
        if key in self._members:
            return self._members[key]
        elif self.is_valid_key(key):
            self._members[key] = self.__class__(self._src_dataset, key)
            return self._members[key]
        raise AttributeError(f"'{key}' is not a valid key for this metric.")

    def __repr__(self):
        if self.key is None:
            return f"{self.__class__.__name__}()"
        return f'{self.__class__.__name__}("{self.key}")'


class WER(Metric):
    """Word Error Rate (WER) metric representation.

    Attributes:
        long_name (str): The long name of the metric.
        short_name (str): The short name of the metric.
        description (str): The description of the metric.
        edits (int): The number of edits (insertions, deletions, substitutions).
        num_ref_tokens (int): The length of the reference text.
    """

    short_name = "WER"
    long_name = "Word Error Rate"
    attr_name = "wer"
    description = (
        "Word Error Rate (WER) is computed as the token-level (i.e., word-level) edit distance between the reference "
        "and hypothesis texts, divided by the total number of tokens in the reference texts."
    )

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits."""
        return sum([example.metrics.wer.num_edits for example in self._src_dataset])

    @cached_property
    def num_ref_tokens(self) -> int:
        """Get the accumulated length of the reference texts."""
        return sum([example.metrics.wer.num_ref_tokens for example in self._src_dataset])

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


class CER(Metric):
    """Character Error Rate (WER) metric representation."""

    short_name = "CER"
    long_name = "Character Error Rate"
    attr_name = "cer"
    description = (
        "Character Error Rate (CER) is computed as the character-level edit distance between the normalized reference "
        "and hypothesis texts, divided by the total number of characters in the reference texts."
    )

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits."""
        return sum([example.metrics.cer.num_edits for example in self._src_dataset])

    @cached_property
    def num_ref_chars(self) -> int:
        """Get the accumulated length of the reference texts."""
        return sum([example.metrics.cer.num_ref_chars for example in self._src_dataset])

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


class VCER(Metric):
    """Vocabulary Error Rate (WER) metric representation.

    Attributes:
        long_name (str): The long name of the metric.
        short_name (str): The short name of the metric.
        description (str): The description of the metric.
        edits (int): The number of edits (insertions, deletions, substitutions).
        num_ref_tokens (int): The length of the reference text.
    """

    short_name = "VCER"
    long_name = "Vocabulary Character Error Rate"
    attr_name = "vcer"
    description = (
        "Vocabulary Character Error Rate (VCER) is computed as the number of character-level edits needed to convert "
        "the reference vocabulary tokens to its aligned hypothesis token, divided by the total number of characters "
        "in the reference vocabulary tokens."
    )

    @cached_property
    def num_edits(self) -> int:
        """Get the number mathed vocabulary tokens."""
        if self.key is None:
            raise ValueError(f"The {self.short_name} metric requires a key: metric['key'].")
        from bewer.alignment.edit_distance import levenshtein_score_matrix
        vocab = self._src_dataset.vocabs[self.key]
        num_edits = 0
        for example in self._src_dataset:
            for token in example.ref.tokens["word"]:
                if token.normalized in vocab:
                    tgt_len = len(token.normalized)
                    D = levenshtein_score_matrix(example.hyp.normalized, token.normalized)
                    offsets = D[:-tgt_len, 0]
                    ends = D[tgt_len:, -1]
                    diffs = ends - offsets
                    int(np.argmin(diffs))
                    import IPython; IPython.embed(using=False)
                    # if token.levenshtein.type == OpType.INSERT:
                    #     num_edits += len(token.normalized)
                    # elif token.levenshtein.type == OpType.SUBSTITUTE:
                    #     num_edits += Levenshtein.distance(token.normalized, token.levenshtein.hyp_token.normalized)

        # predictions = pred.predictions[0].lower()
        # medical_terms = [term.lower() for term in pred.medical_terms[0]]

        # total_distance = 0
        # total_length = 0

        # if predictions != "":
        #     for term in medical_terms:
        #         n = len(term.split())
        #         ngrams = generate_ngrams(predictions, n)

        #         best_match, _ = process.extractOne(term, ngrams, scorer=fuzz.ratio)
        #         distance = levenshtein_distance(term, best_match)

        #         total_distance += distance
        #         total_length += max(len(term), 1)

        # cer_keyword = total_distance / total_length if total_length > 0 else 0

        # return cer_keyword, total_distance, total_length
        return num_edits

    @cached_property
    def num_vocab_chars(self) -> int:
        """Get the number of vocabulary tokens in the references."""
        if self.key is None:
            raise ValueError(f"The {self.short_name} metric requires a key: metric['key'].")
        vocab = self._src_dataset.vocabs[self.key]
        num_vocab_chars = 0
        for example in self._src_dataset:
            for token in example.ref.tokens["word"]:
                if token.normalized in vocab:
                    num_vocab_chars += len(token.normalized)
        return num_vocab_chars

    @cached_property
    def value(self) -> float:
        if self.num_vocab_chars == 0:
            return 0.0
        return self.num_edits / self.num_vocab_chars

    @cached_property
    def rvalue(self) -> float:
        return round(self.value, 4)

    @cached_property
    def fvalue(self) -> str:
        return f"{self.value * 100:.2f}%"

    def is_valid_key(self, key: str) -> bool:
        if key in self._src_dataset.vocabs.keys():
            return True
        return False


class VR(Metric):
    """Vocabulary Recall (VR) metric representation.

    Attributes:
        long_name (str): The long name of the metric.
        short_name (str): The short name of the metric.
        description (str): The description of the metric.
        edits (int): The number of edits (insertions, deletions, substitutions).
        num_ref_tokens (int): The length of the reference text.
    """

    short_name = "VR"
    long_name = "Vocabulary Recall"
    attr_name = "vr"
    description = (
        "Vocabulary Recall (VR) is computed as the number of correctly transcribed vocabulary tokens, divided by the "
        "total number of vocabulary tokens in the reference texts."
    )

    @cached_property
    def num_matches(self) -> int:
        """Get the number mathed vocabulary tokens."""
        if self.key is None:
            raise ValueError(f"The {self.short_name} metric requires a key: metric['key'].")
        vocab = self._src_dataset.vocabs[self.key]
        num_matches = 0
        for example in self._src_dataset:
            for op in example.levenshtein.ops["match"]:
                if op.ref_token.normalized in vocab:
                    num_matches += 1
        return num_matches

    @cached_property
    def num_vocab_tokens(self) -> int:
        """Get the number of vocabulary tokens in the references."""
        if self.key is None:
            raise ValueError(f"The {self.short_name} metric requires a key: metric['key'].")
        vocab = self._src_dataset.vocabs[self.key]
        num_vocab_tokens = 0
        for example in self._src_dataset:
            for token in example.ref.tokens["word"]:
                if token.normalized in vocab:
                    num_vocab_tokens += 1
        return num_vocab_tokens

    @cached_property
    def value(self) -> float:
        if self.num_vocab_tokens == 0:
            return 0.0
        return self.num_matches / self.num_vocab_tokens

    @cached_property
    def rvalue(self) -> float:
        return round(self.value, 4)

    @cached_property
    def fvalue(self) -> str:
        return f"{self.value * 100:.2f}%"

    def is_valid_key(self, key: str) -> bool:
        if key in self._src_dataset.vocabs.keys():
            return True
        return False


# class PuER(Metric):
#     """Punctuation Error Rate (PuER) metric representation.

#     Attributes:
#         long_name (str): The long name of the metric.
#         short_name (str): The short name of the metric.
#         description (str): The description of the metric.
#         edits (int): The number of edits (insertions, deletions, substitutions).
#         num_ref_tokens (int): The length of the reference text.
#     """

#     short_name = "PuER"
#     long_name = "Punctuation Error Rate"
#     attr_name = "puer"
#     description = "Punctuation Error Rate (WER) computes the ..."

#     @cached_property
#     def edits(self) -> int:
#         """Get the number of punctuation edits while accounting for inserted and deleted tokens."""
#         punctuation_edits = 0
#         lookahead_delete, lookahead_insert = None, None
#         for example in self._src_dataset:
#             for op in example.levenshtein.ops:

#                 # Resolve outstanding lookahead for deletions
#                 if lookahead_delete is not None:
#                     if op.type == OpType.DELETE:
#                         if op.hyp_token.punctuation == lookahead_delete:
#                             lookahead_delete = None
#                             continue
#                         elif op.hyp_token.punctuation is None:
#                             continue
#                     punctuation_edits += 1
#                     lookahead_delete = None

#                 # Resolve outstanding lookahead for insertions
#                 if lookahead_insert is not None:
#                     if op.type == OpType.INSERT:
#                         if op.ref_token.punctuation == lookahead_insert:
#                             lookahead_insert = None
#                             continue
#                         elif op.ref_token.punctuation is None:
#                             continue
#                     punctuation_edits += 1
#                     lookahead_insert = None

#                 # Case 1: Mismatching punctuation for match or substitute
#                 if op.type in (OpType.MATCH, OpType.SUBSTITUTE):
#                     if op.ref_token.punctuation == op.hyp_token.punctuation:
#                         continue
#                     elif op.ref_token.punctuation is None:
#                         lookahead_delete, lookahead_insert = None, op.hyp_token.punctuation
#                     elif op.hyp_token.punctuation is None:
#                         lookahead_delete, lookahead_insert = op.ref_token.punctuation, None
#                     else:
#                         punctuation_edits += 1
#                 # Case 2: Inserted punctuation token
#                 elif op.type == OpType.INSERT and op.ref_token.punctuation is not None:
#                     punctuation_edits += 1
#                 # Case 3: Deleted punctuation token
#                 elif op.type == OpType.DELETE and op.hyp_token.punctuation is not None:
#                     punctuation_edits += 1

#         return punctuation_edits

#     @cached_property
#     def num_ref_tokens(self) -> int:
#         """Get the number of punctuation tokens in the references."""
#         return len(self._src_dataset.refs.tokens["punctuation"])

#     @cached_property
#     def value(self) -> float:
#         if self.num_ref_tokens == 0:
#             return 0.0
#         return self.edits / self.num_ref_tokens

#     @cached_property
#     def rvalue(self) -> float:
#         return round(self.value, 4)

#     @cached_property
#     def fvalue(self) -> str:
#         return f"{self.value * 100:.2f}%"
