from typing import List

def split_into_chunks(text: str,
                      max_sentences: int = 6,
                      delimiter: str = '\n',
                      max_chars: int | None = None) -> List[str]:

    # TODO: sentence_transformers attention is 256 tokens. 
    # 6 sentences can have more than that no checking or limiting currently.
    # TODO: max_chars defaults to 1000 which resolves the above issue. 
    # but for different transformers, this may not be the case. 
    # Need to check for max tokens for the transformer and limit accordingly.
    # TODO: max_chars is implemented which takes into account tokens by embedder
    # tested for sentence_embedding only
    paragraphs = [p for p in text.split(delimiter) if p.strip()]
    chunks = []
    buffer, buffer_len = [], 0
    for p in paragraphs:
        p = p.strip()
        if buffer and (len(buffer) >= max_sentences
                       or (max_chars is not None and buffer_len + len(p) > max_chars)):
            chunks.append(' '.join(buffer))
            buffer = []
            buffer_len = 0
        buffer.append(p)
        buffer_len += len(p)
        # buffer.append(p.strip())
        # if len(buffer) >= max_sentences:
        #     chunks.append(' '.join(buffer))
        #     buffer = []
    if buffer:
        chunks.append(' '.join(buffer))
    return chunks