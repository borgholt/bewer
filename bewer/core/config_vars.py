from contextvars import ContextVar


TEXT_NORMALIZER = ContextVar("text_normalizer", default=None)
TOKEN_NORMALIZER = ContextVar("token_normalizer", default=None)
TOKENIZER = ContextVar("tokenizer", default=None)