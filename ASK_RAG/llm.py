import numpy as np
import os
import logging
import json
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from .utils import (
    wrap_embedding_func_with_attrs
)

logger = logging.getLogger("ASK_RAG")

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url="https://api.openai.com/v1",
    api_key="",
) -> np.ndarray:
    if api_key:
        os.environ["LLM_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def call_llm_for_keywords(client: AsyncOpenAI, model: str, prompt: str) -> dict | None:
    """Calls the LLM for keyword extraction and parses the JSON response."""
    logger.debug(f"Calling model {model} for keyword extraction...")
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             logger.error("LLM response is empty or invalid.")
             return None
             
        content = response.choices[0].message.content
        logger.debug(f"LLM Raw JSON Response: {content}")

        try:
            keyword_dict = json.loads(content)
            if not isinstance(keyword_dict, dict):
                logger.error(f"LLM response is not a valid JSON dictionary: {content}")
                return None
            return keyword_dict
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON response from LLM: {json_err}")
            logger.error(f"Problematic Content: {content}")
            return None

    except Exception as e:
        logger.error(f"Error calling OpenAI API during keyword extraction: {e}", exc_info=True)
        raise e
    
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def call_llm_for_edge_prediction(client: AsyncOpenAI, model: str, prompt: str) -> str | None:
    """Calls the LLM to generate a textual description for a predicted edge based on the provided prompt."""
    logger.debug(f"Calling model {model} for edge description generation with prompt:\n{prompt[:200]}...")
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, 
        )

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.error("LLM response is empty or invalid for edge description.")
            return None
            
        content = response.choices[0].message.content.strip()
        logger.debug(f"LLM Raw Text Response for edge description: {content}")

        if not content:
            logger.error("LLM response content is effectively empty after stripping.")
            return None
            
        return content

    except Exception as e:
        logger.error(f"Error calling OpenAI API during edge description generation: {e}", exc_info=True)
        raise e