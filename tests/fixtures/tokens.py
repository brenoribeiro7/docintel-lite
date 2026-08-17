import tiktoken

from docintel.indexing.chunking import CHUNK_ENCODING

ENCODING = tiktoken.get_encoding(CHUNK_ENCODING)
REPEATABLE_TOKEN = ENCODING.encode(" a")
assert len(REPEATABLE_TOKEN) == 1


def text_with_token_count(count: int) -> str:
    return ENCODING.decode(REPEATABLE_TOKEN * count)
