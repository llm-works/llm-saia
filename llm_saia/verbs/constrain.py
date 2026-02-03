"""CONSTRAIN verb: Parse response into structured schema."""

from typing import TypeVar

from llm_saia.core.protocols import SAIABackend

T = TypeVar("T")


async def constrain(backend: SAIABackend, response: str, schema: type[T]) -> T:
    """CONSTRAIN verb: Parse response into structured schema.

    Args:
        backend: The LLM backend to use.
        response: The unstructured response to parse.
        schema: A dataclass type to parse the response into.

    Returns:
        An instance of the schema type populated from the response.
    """
    prompt = f"""Parse the following response into the requested structured format.
Extract the relevant information and format it according to the schema.

Response to parse:
{response}

Extract and structure the information."""
    return await backend.complete_structured(prompt, schema)
