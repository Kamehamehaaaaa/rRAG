from typing import List

def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return [
        ". ".join(sentences[i:i + chunk_size]) + "."
        for i in range(0, len(sentences), chunk_size)
    ]