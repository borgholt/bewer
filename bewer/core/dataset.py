import io
import ast
from collections.abc import Iterable
from functools import cached_property
from itertools import chain

import pandas as pd
from mlflow.artifacts import load_text

from bewer.core.example import Example
from bewer.core.text import TextList
from bewer.core.op import OpList
from bewer.metrics.base import MetricCollection
from bewer.preprocessing.tokenization import Tokenizer
from bewer.preprocessing.normalization import TextNormalizer
from bewer.preprocessing.normalization import TokenNormalizer
from bewer.core.config_vars import TOKEN_NORMALIZER
from bewer.core.config_vars import TEXT_NORMALIZER
from bewer.core.config_vars import TOKENIZER


def is_list_literal(s):
    try:
        return isinstance(ast.literal_eval(s), list)
    except (ValueError, SyntaxError):
        return False


class Dataset(object):
    """BeWER dataset representation.

    Attributes:
        config (...): ...
        examples (list[Example]): A list of Example objects.
        metrics (MetricCollection): A collection of metrics for the dataset.
        refs (TextList): The reference texts in the dataset.
        hyps (TextList): The hypothesis texts in the dataset.
        ops (OpList): The operations in the dataset.
    """

    def __init__(self, config=None):

        self.config = config
        self.examples = []
        self.vocabs = {}
        self.metrics = MetricCollection(self)
        self._inference_function = None
        self._locked = False

        if config is None:
            TOKENIZER.set(Tokenizer())
            TOKEN_NORMALIZER.set(TokenNormalizer())
            TEXT_NORMALIZER.set(TextNormalizer())

    @cached_property
    def refs(self) -> TextList:
        """Get the reference texts as a TextList object.

        Returns:
            TextList: The reference texts.
        """
        return TextList([example.ref for example in self.examples])

    @cached_property
    def hyps(self) -> TextList:
        """Get the hypothesis texts as a TextList object.

        Returns:
            TextList: The hypothesis texts.
        """
        return TextList([example.hyp for example in self.examples])

    @cached_property
    def ops(self) -> OpList:
        """Get the operations as an OpList object.

        Returns:
            OpList: The operations.
        """
        ops = OpList([])
        for example in self.examples:
            ops.extend(example.levenshtein.ops)
        return ops

    def add(self, ref: str, hyp: str) -> None:
        """Add an example to the dataset."""
        if self._locked:
            raise RuntimeError("Dataset is locked. Cannot add more examples.")
        example = Example(ref, hyp, _src_dataset=self)
        self.examples.append(example)

    def add_vocab(self, name: str, vocab: Iterable[str]) -> None:
        """Add vocabulary items and update tokenizer.

        Args:
            name (str): The name of the vocabulary.
            vocab (Iterable[str]): An iterable of vocabulary items.
        """
        assert all(isinstance(item, str) for item in vocab), "Vocabulary items must be strings"
        vocab = set(map(str.lower, vocab))
        tokenizer = TOKENIZER.get()
        tokenizer.add_vocab_items(vocab)
        self.vocabs[name] = vocab

    def set_inference_function(self, inference_function) -> None:
        """Set the inference function for the dataset."""
        if not callable(inference_function):
            raise TypeError("inference_function must be callable")
        self._inference_function = inference_function

    def _infer_column_vocab(self, series: pd.Series) -> set:
        """Infer the vocabulary from a pandas Series."""
        if series.map(is_list_literal).all():
            series = series.apply(ast.literal_eval)
            return series.name, set(chain(*series))
        elif series.map(lambda x: isinstance(x, str)).all():
            return series.name, series
        elif series.map(lambda x: isinstance(x, list)).all():
            return series.name, set(chain(*series))
        else:
            raise ValueError(f"Column {series.name} is not a list or string")

    def load_dataset(self, dataset, ref_col="ref", hyp_col="hyp", vocab_cols: list = []) -> None:
        """Load a Hugging Face dataset."""
        raise NotImplementedError("load_dataset() method not implemented.")

    def load_pandas(self, df, ref_col="ref", hyp_col="hyp", vocab_cols: list = []) -> None:
        """Add a pandas DataFrame to the dataset."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        # Prepare and add vocabulary phrases to the tokenizer
        for col in vocab_cols:
            self.add_vocab(*self._infer_column_vocab(df[col]))

        # Add examples to the dataset
        for row in df.itertuples(index=False):
            hyp = getattr(row, hyp_col)
            ref = getattr(row, ref_col)
            self.add(ref, hyp)
        return self

    def load_csv(self, csv_file: str, ref_col="ref", hyp_col="hyp", vocab_cols: list = []) -> None:
        """Add a CSV file to the dataset."""
        df = pd.read_csv(csv_file)
        self.load_pandas(df, ref_col, hyp_col, vocab_cols)
        return self

    def load_mlflow_csv(self, csv_uri: str, ref_col="ref", hyp_col="hyp", vocab_cols: list = []) -> None:
        """Add a CSV file from MLflow to the dataset."""
        csv_str = load_text(csv_uri)
        csv_buffer = io.StringIO(csv_str)
        self.load_csv(csv_buffer, ref_col, hyp_col, vocab_cols)
        return self

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        """Get an example by index."""
        return self.examples[index]

    def __iter__(self):
        """Iterate over the examples in the dataset."""
        return iter(self.examples)

    def __repr__(self):
        """Get a string representation of the dataset."""
        return f"Dataset({len(self.examples)} examples)"
