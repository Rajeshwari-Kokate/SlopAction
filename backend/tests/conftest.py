"""Shared test fixtures and sample texts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

TINY_MODELS = BACKEND_ROOT / "tests" / "fixtures" / "models"


# --------------------------------------------------------------------------
# sample texts
# --------------------------------------------------------------------------

SHORT_TEXT = "Hello how are you?"

ONE_SENTENCE = (
    "The committee approved the revised budget after a two-hour debate that "
    "ran well past the scheduled end of the session on Thursday afternoon.")

HUMAN_LIKE = """I spent most of Saturday trying to fix the boiler. It's an old
thing, installed sometime in the nineties by a plumber who clearly had opinions
about pipe routing. Two hours in, I found the problem: a corroded diverter
valve, seized solid.

My neighbour Dave wandered over, looked at it, and said "that's a Baxi, that
is." He was wrong. It's a Potterton. But he did lend me a wrench, which turned
out to matter more than the diagnosis.

Anyway. New valve ordered, £48 plus delivery, arriving Tuesday. Until then
we're boiling kettles like it's 1974. The kids think this is hilarious. I do
not think this is hilarious. My wife has taken to describing the situation as
"characterful", which is the word she uses when she means "your fault".

I'll get it sorted before the cold snap. Probably."""

AI_LIKE = """Artificial intelligence is fundamentally transforming the landscape
of modern education. In today's fast-paced world, machine learning tools play a
crucial role in classrooms across the globe. Furthermore, adaptive learning
platforms personalise instruction for each individual student, enabling
educators to tailor content effectively.

Moreover, artificial intelligence is changing how education fundamentally
works. Education is being revolutionised by artificial intelligence systems in
numerous ways. Consequently, schools must adapt their approaches to remain
relevant in an increasingly digital environment.

Additionally, the integration of these tools requires careful consideration.
Institutions should generally approach adoption thoughtfully and deliberately.
Teachers may potentially need additional training and support to leverage these
technologies effectively.

In conclusion, the future of learning depends on thoughtful adoption of
artificial intelligence. To summarise, schools that embrace these tools while
maintaining human connection will thrive. Ultimately, the key to success lies in
striking the right balance."""

MIXED_TEXT = HUMAN_LIKE + "\n\n" + AI_LIKE

CODE_TEXT = """```python
def compute_perplexity(model, tokenizer, text):
    import torch
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


class Detector:
    def __init__(self, name):
        self.name = name

    def run(self, text):
        return {"name": self.name, "score": None}
```
const parse = (raw) => JSON.parse(raw);
import numpy as np
SELECT id, name FROM users WHERE active = 1;
"""

POETRY_TEXT = """The light retreats and so we know it now
the season turns its face against the wall
a slow undoing, quieter than a vow
and colder than the hallway after all

We counted out the summer in the yard
in bottle caps and bruises on the knee
the gate stood open, then the gate stood barred
and nobody explained the change to me

I keep the kettle warm against the draught
and listen to the pipes rehearse their song
a fractured, tuneless, half-remembered draft
of something that I hummed when I was young"""

MATH_TEXT = r"""Let $f: \mathbb{R}^n \to \mathbb{R}$ be twice differentiable.
\begin{equation}
\nabla^2 f(x) = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}
\end{equation}
Then $\alpha \leq \beta$ implies $f(\alpha) \leq f(\beta)$ when $f' \geq 0$.
We have $x^2 + y^2 = r^2$ and $\int_0^1 x^n dx = 1/(n+1)$.
Therefore $\sum_{k=1}^{\infty} 1/k^2 = \pi^2/6$ by Euler's identity.
\[ E = mc^2 \quad \text{and} \quad \theta \in [0, 2\pi) \]"""

NON_ENGLISH = """Die Entwicklung der künstlichen Intelligenz hat in den letzten
Jahren erhebliche Fortschritte gemacht. Insbesondere im Bereich der
Sprachverarbeitung wurden bemerkenswerte Ergebnisse erzielt. Die Modelle können
inzwischen Texte erzeugen, die von menschlichen Texten kaum zu unterscheiden
sind. Dies wirft neue Fragen für Bildungseinrichtungen und Verlage auf, die sich
mit der Frage der Urheberschaft befassen müssen. Forscher arbeiten daher an
Verfahren zur Erkennung maschinell erzeugter Inhalte."""

EMOJI_TEXT = ("omg this is actually wild 😂😂 " * 12) + \
    "no way 🔥🔥🔥 can't believe it tbh 💀 lowkey the best thing all week ngl 🙌"

URL_TEXT = ("See https://example.com/docs/getting-started and "
            "https://another.example.org/page?q=1 for details. Contact "
            "someone@example.com or team@example.org. " * 8)

SPECIAL_CHARS = ("Testing «quotes» and — dashes… plus ‹these› and „those‟ "
                 "with ±±± symbols ©®™ and ½¾ fractions ∑∏∫ math ✓✗ marks. " * 8)

LONG_TEXT = (AI_LIKE + "\n\n") * 12


@pytest.fixture(scope="session")
def human_text() -> str:
    return HUMAN_LIKE


@pytest.fixture(scope="session")
def ai_text() -> str:
    return AI_LIKE


@pytest.fixture(scope="session")
def mixed_text() -> str:
    return MIXED_TEXT


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Detector caches are process-level by design; clear the memoised
    ``functools.lru_cache`` entries that depend on artefact presence so tests
    cannot leak state into each other."""
    yield
    try:
        from app.detectors import normalisation
        from app.preprocessing import category

        normalisation._load.cache_clear()
        category._trained_model.cache_clear()
    except Exception:  # pragma: no cover
        pass


@pytest.fixture(scope="session")
def tiny_models_available() -> bool:
    return (TINY_MODELS / "causal_lm").exists()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "smoke: exercises neural code paths with the tiny randomly-initialised "
        "fixture models. Outputs are meaningless; only plumbing is asserted.")
    config.addinivalue_line(
        "markers", "slow: takes more than a couple of seconds")
