from functools import partial

import regex as re

from bewer.preprocessing.tokenization import Token


def default_token_normalizer(text: str) -> str:
    """
    Basic normalizer that lowercases the input text and strips leading/trailing whitespace.

    Args:
        text (str): Input token string.

    Returns:
        str: Normalized token.
    """

    return text.lower().strip()

def default_text_normalizer(text: str) -> str:
    """
    Basic text normalizer that lowercases the input text.

    Args:
        text (str): Input text string.

    Returns:
        str: Normalized text.
    """    
    return text.lower()


class TokenNormalizer(object):

    def __init__(self, func: callable = default_token_normalizer, **normalizer_kwargs):
        """Initialize a token-level normalizer with a function.

        Args:
            func (callable): The normalization function.
            normalizer_kwargs (dict): Additional keyword arguments for the normalization function.
        """
        self.func = partial(func, **normalizer_kwargs)

    def __call__(self, token: str) -> list[Token]:
        """Normalize the input token using the normalization function.

        Args:
            token (str): The input token to normalize.

        Returns:
            str: The normalized token.
        """
        return self.func(token)

    def __repr__(self):
        return f"TokenNormalizer({self.func.func.__name__})"


class TextNormalizer(object):

    def __init__(self, func: callable = default_text_normalizer, **normalizer_kwargs):
        """Initialize a text-level normalizer with a function.

        Args:
            func (callable): The normalization function.
            normalizer_kwargs (dict): Additional keyword arguments for the normalization function.
        """
        self.func = partial(func, **normalizer_kwargs)

    def __call__(self, text: str) -> list[Token]:
        """Normalize the input text using the normalization function.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        return self.func(text)
    
    def __repr__(self):
        return f"TextNormalizer({self.func.func.__name__})"
