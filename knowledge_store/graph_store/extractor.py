from . import AbstractExtractor
import config
from .schema import ExtractedGraph
from .registry import register

class OllamaGraphExtractor(AbstractExtractor):
    def __init__(self, model: str):
        self.model = model


    def extract(self, text: str) -> ExtractedGraph:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[{
                'role': 'system',
                'content': ( 
                    'You are an expert AI extraction tool. '
                    'Your job is to extract a knowledge graph from the given text input. '
                    'Strictly adhere to the provided JSON schema and '
                    'give explainations in description field if any.'
                )
            },
            {
                "role": "user",
                "content": (
                    "Extract components and relationships from this text as a graph, "
                    "for a technical knowledge base. If the text describes no clear "
                    "components or relationships, return empty lists. Do not invent "
                    "structure that isn't there.\n\n" + text
                ),
            }],
            format=ExtractedGraph.model_json_schema(),
        )
        return ExtractedGraph.model_validate_json(response["message"]["content"])


register("ollama", lambda: OllamaGraphExtractor(model=config.OLLAMA_MODEL))