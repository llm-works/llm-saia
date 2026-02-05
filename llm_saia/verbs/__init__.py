"""SAIA verb implementations."""

from llm_saia.core.types import VerbConfig
from llm_saia.core.verb import Verb
from llm_saia.verbs.ask import Ask
from llm_saia.verbs.choose import Choose
from llm_saia.verbs.classify import Classify
from llm_saia.verbs.complete import Complete
from llm_saia.verbs.confirm import Confirm
from llm_saia.verbs.constrain import Constrain
from llm_saia.verbs.critique import Critique_
from llm_saia.verbs.decompose import Decompose
from llm_saia.verbs.extract import Extract
from llm_saia.verbs.ground import Ground
from llm_saia.verbs.instruct import Instruct
from llm_saia.verbs.memory import recall, store
from llm_saia.verbs.refine import Refine
from llm_saia.verbs.synthesize import Synthesize
from llm_saia.verbs.verify import Verify

__all__ = [
    # Base
    "VerbConfig",
    "Verb",
    # Verb classes
    "Ask",
    "Choose",
    "Classify",
    "Complete",
    "Confirm",
    "Constrain",
    "Critique_",
    "Decompose",
    "Ground",
    "Instruct",
    "Extract",
    "Refine",
    "Synthesize",
    "Verify",
    # Memory functions (non-LLM)
    "recall",
    "store",
]
