"""
src/llm_provider.py
Modular LLM factory supporting multiple providers:
  - 'langchain_openai': ChatOpenAI from langchain_openai (custom base_url, api_key, model)
  - 'openai': Direct OpenAI client
  - 'offline': Fallback when no keys/providers are configured
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm_client(provider: str = None):
    """
    Initialize and return an LLM client or wrapper based on the specified provider.

    Args:
        provider (str): 'langchain_openai', 'openai', or 'offline'.
                       Defaults to the LLM_PROVIDER env variable or 'openai'.

    Returns:
        tuple: (client_instance, provider_type) or (None, 'offline')
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    else:
        provider = provider.lower().strip()

    # --- Option 1: LangChain ChatOpenAI ---
    if provider in ("langchain_openai", "langchain", "chatopenai"):
        try:
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("LANGCHAIN_OPENAI_API_KEY") or os.getenv(
                "OPENAI_API_KEY", ""
            )
            base_url = os.getenv("LANGCHAIN_OPENAI_BASE_URL") or os.getenv(
                "OPENAI_BASE_URL", None
            )
            model_name = os.getenv("LANGCHAIN_OPENAI_MODEL_NAME") or os.getenv(
                "OPENAI_MODEL_NAME", "gpt-4.1"
            )

            # Check if key is configured
            if not api_key or "your_" in api_key:
                logger.warning(
                    "No valid API key found for langchain_openai. Defaulting to offline mode."
                )
                return None, "offline"

            client_kwargs = {
                "model": model_name,
                "api_key": api_key,
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            }
            if base_url and base_url.strip():
                client_kwargs["base_url"] = base_url.strip()

            llm = ChatOpenAI(**client_kwargs)
            logger.info(
                f"Initialized LangChain ChatOpenAI with model: {model_name}"
            )
            return llm, "langchain_openai"

        except ImportError:
            logger.error(
                "langchain_openai is not installed. Run: pip install langchain-openai"
            )
            return None, "offline"
        except Exception as e:
            logger.error(f"Error initializing LangChain ChatOpenAI: {e}")
            return None, "offline"

    # --- Option 2: Default Direct OpenAI SDK ---
    elif provider in ("openai", "default"):
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL", None)

            if not api_key or "your_" in api_key:
                logger.warning(
                    "No valid API key found for direct OpenAI. Defaulting to offline mode."
                )
                return None, "offline"

            client_kwargs = {"api_key": api_key}
            if base_url and base_url.strip():
                client_kwargs["base_url"] = base_url.strip()

            client = OpenAI(**client_kwargs)
            logger.info("Initialized direct OpenAI client.")
            return client, "openai"

        except ImportError:
            logger.error(
                "openai package not installed. Run: pip install openai"
            )
            return None, "offline"
        except Exception as e:
            logger.error(f"Error initializing direct OpenAI: {e}")
            return None, "offline"

    # --- Option 3: Explicit Offline / Unknown ---
    else:
        logger.info(f"Provider '{provider}' set to offline mode.")
        return None, "offline"


def generate_llm_response(prompt: str) -> str:
    """
    Unified caller that routes the prompt through whichever LLM provider is active.

    Args:
        prompt (str): Text prompt to send to the LLM.

    Returns:
        str: Response text from the model.
    """
    client, provider_type = get_llm_client()

    if provider_type == "offline" or client is None:
        return ""

    if provider_type == "langchain_openai":
        # LangChain ChatOpenAI returns an AIMessage object with .content
        response = client.invoke(prompt)
        return response.content.strip()

    elif provider_type == "openai":
        # Direct OpenAI client completion
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return response.choices[0].message.content.strip()

    return ""
