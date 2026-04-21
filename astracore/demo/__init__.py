"""``astracore demo`` — run real inference with sensible inputs.

The zoo benchmark (``astracore zoo``) measures **speed**. The demo
(``astracore demo``) measures **correctness-of-output**: feed a real
image / prompt, run inference, decode the output into something a
human can recognise ("top-5: school bus 42.3%"). This is what OEM
evaluators actually look at first.

Per-family dispatch:
  * vision-classification → ImageNet top-5
  * vision-detection      → YOLOv8 bounding boxes
  * nlp-encoder-transformer → BERT-Squad answer-span (canned input)
  * nlp-decoder-transformer → GPT-2 next-token logits (canned prompt)

All demos are also OEM-pluggable — a custom zoo entry with a new
``family`` string can register its own handler via
``@register_demo_family("my-family")``.
"""

from astracore.demo.base import (
    DemoResult,
    DemoError,
    register_demo_family,
    get_demo_handler,
    run_demo,
)

__all__ = [
    "DemoResult",
    "DemoError",
    "register_demo_family",
    "get_demo_handler",
    "run_demo",
]
