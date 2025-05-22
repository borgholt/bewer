from typing import Optional
from typing import TYPE_CHECKING
from functools import cached_property

from bewer.metrics.base import MetricCollection
from bewer.core.alignment import LevenshteinAlignment
from bewer.core.alignment import SpanAlignment
from bewer.core.text import Text

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset


class Example:
    """
    BeWER example representation.

    Attributes:
        ref (Text): Reference text object.
        hyp (Text): Hypothesis text object.
        levenshtein (LevenshteinAlignment): Levenshtein alignment object.
    """

    def __init__(self, ref: str, hyp: str, _src_dataset: Optional["Dataset"] = None):
        """
        Initialize the Example object.

        Args:
            ref (str): Reference text.
            hyp (str): Hypothesis text.
        """
        self._src_dataset = _src_dataset
        self._inference_metrics = None
        self.ref = Text(ref, _src_example=self, _is_hyp=False)
        self.hyp = Text(hyp, _src_example=self, _is_hyp=True)
        self.metrics = MetricCollection(self)

    @cached_property
    def levenshtein(self) -> LevenshteinAlignment:
        """
        Get the Levenshtein alignment object.

        Returns:
            LevenshteinAlignment: The Levenshtein alignment object.
        """
        return LevenshteinAlignment(self)
    
    # @cached_property
    @property
    def spanalign(self) -> SpanAlignment:
        """
        Get the SpanAligner alignment object.

        Returns:
            LevenshteinAlignment: The SpanAligner alignment object.
        """
        return SpanAlignment(self)
    

    @cached_property
    def inference_metrics(self) -> Optional[dict]:
        """
        Get the inference metrics.

        Returns:
            dict: The inference metrics.
        """
        if not isinstance(self.hyp.raw, str):
            raise ValueError("Couldn't get inference metrics. The text is malformed.")
        return self._inference_metrics

    def __repr__(self):
        ref = self.ref.raw if len(self.ref.raw) <= 45 else self.ref.raw[:42] + "..."
        hyp = self.hyp.raw if len(self.hyp.raw) <= 45 else self.hyp.raw[:42] + "..."
        return f'Example(ref="{ref}", hyp="{hyp}")'
