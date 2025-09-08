# gemini_sre_agent/llm/__init__.py

"""
Multi-LLM Provider Support System

This module provides a unified interface for multiple LLM providers including
Gemini, Ollama, Claude, ChatGPT, Grok, and Amazon Bedrock with advanced
model mixing capabilities, cost optimization, and enterprise-grade resilience.
"""

from .base import (
    CircuitBreaker,
    ErrorSeverity,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)
from .common.enums import ModelType, ProviderType
from .factory import LLMProviderFactory
from .providers import (
    AnthropicProvider,
    BedrockProvider,
    GeminiProvider,
    GrokProvider,
    OllamaProvider,
    OpenAIProvider,
)

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ModelType",
    "ProviderType",
    "ErrorSeverity",
    "LLMProviderError",
    "CircuitBreaker",
    "LLMProviderFactory",
    # Concrete providers
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "GrokProvider",
    "BedrockProvider",
]
