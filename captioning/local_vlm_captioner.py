from . import AbstractCaptioner
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Describe this technical diagram for a search index. Include:
- What process or system it depicts
- Key components and their count (motors, belts, valves, sensors, etc.)
- How components relate or connect (flow, sequence, hierarchy)
- Any visible labels, numbers, or text — transcribe them exactly

{context_block}
Be specific and factual. Do not guess at values you can't clearly see."""


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class LocalVLMCaptioner(AbstractCaptioner):
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl", device: Optional[str] = None):
        self.device = device if device is not None else _resolve_device()
        logger.info("loading captioning model %s on %s", model_name, self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=self.device if self.device == "cuda" else None,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def caption(self, image_path: Path, context: Optional[str] = None) -> str:
        image = Image.open(image_path).convert("RGB")
        context_block = f"Surrounding manual text for context: {context}\n" if context else ""
        prompt = _PROMPT_TEMPLATE.format(context_block=context_block)

        result = self.model.query(image, prompt)
        return result["answer"].strip()