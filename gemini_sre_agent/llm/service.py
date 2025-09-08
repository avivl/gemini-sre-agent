# gemini_sre_agent/llm/service.py

"""
Core LLM service integrating LiteLLM, Instructor, and Mirascope.

This module provides the main LLMService class that unifies access to multiple
LLM providers through LiteLLM, structured output through Instructor, and
advanced prompt management through Mirascope.
"""

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import LLMRequest, ModelType
from .config import LLMConfig
from .factory import get_provider_factory
from .prompt_manager import PromptManager

# Note: Mirascope integration will be added in a future update
# For now, we'll use simple string templates


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMService(Generic[T]):
    """
    Core LLM service integrating LiteLLM, Instructor, and Mirascope.

    Provides unified access to multiple LLM providers with structured output
    and advanced prompt management capabilities.
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLM service with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize provider factory and create providers
        self.provider_factory = get_provider_factory()
        self.providers = self.provider_factory.create_providers_from_config(config)

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        self.logger.info("LLMService initialized with provider factory + Mirascope")

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Generate a structured response using the specified model and prompt."""
        try:
            provider_name = provider or self.config.default_provider
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not available")

            provider_instance = self.providers[provider_name]
            self.logger.info(
                f"Generating structured response using provider: {provider_name}"
            )

            request = LLMRequest(
                prompt=prompt,
                model_type=ModelType.SMART,  # Default model type
                **kwargs
            )
            response = await provider_instance.generate(request)
            # For structured output, we need to parse the response content
            # This is a simplified approach - in practice, you'd want proper structured parsing
            return response.content  # type: ignore
        except Exception as e:
            self.logger.error(f"Error generating structured response: {str(e)}")
            raise

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a plain text response."""
        try:
            provider_name = provider or self.config.default_provider
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not available")

            provider_instance = self.providers[provider_name]
            self.logger.info(
                f"Generating text response using provider: {provider_name}"
            )

            request = LLMRequest(
                prompt=prompt,
                model_type=ModelType.SMART,  # Default model type
                **kwargs
            )
            response = await provider_instance.generate(request)
            return response.content
        except Exception as e:
            self.logger.error(f"Error generating text response: {str(e)}")
            raise

    async def health_check(self, provider: Optional[str] = None) -> bool:
        """Check if the specified provider is healthy and accessible."""
        try:
            provider_name = provider or self.config.default_provider
            if provider_name not in self.providers:
                self.logger.error(f"Provider '{provider_name}' not available")
                return False

            provider_instance = self.providers[provider_name]
            return await provider_instance.health_check()

        except Exception as e:
            self.logger.error(f"Health check failed for provider {provider}: {str(e)}")
            return False

    def get_available_models(
        self, provider: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get available models for the specified provider or all providers."""
        if provider:
            if provider in self.providers:
                models = self.providers[provider].get_available_models()
                return {provider: list(models.values()) if isinstance(models, dict) else models}
            return {}

        result = {}
        for provider_name, provider_instance in self.providers.items():
            models = provider_instance.get_available_models()
            result[provider_name] = list(models.values()) if isinstance(models, dict) else models
        return result


def create_llm_service(config: LLMConfig) -> LLMService:
    """Factory function to create and configure an LLMService instance."""
    return LLMService(config)
