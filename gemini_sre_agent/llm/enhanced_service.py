# gemini_sre_agent/llm/enhanced_service.py

"""
Enhanced LLM Service with Intelligent Model Selection.

This module provides an enhanced LLM service that integrates the semantic model
selection system with the existing provider interfaces, enabling intelligent
model selection based on task requirements, performance metrics, and fallback chains.
"""

import logging
import time
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

try:
    from mirascope.llm import Provider
except ImportError:
    # Provider class not available in current mirascope version
    Provider = None  # type: ignore

from pydantic import BaseModel

from .base import LLMRequest, ModelType
from .common.enums import ProviderType
from .config import LLMConfig
from .factory import get_provider_factory
from .model_registry import ModelInfo, ModelRegistry
from .model_scorer import ModelScorer, ScoringContext, ScoringWeights
from .model_selector import (
    ModelSelector,
    SelectionCriteria,
    SelectionResult,
    SelectionStrategy,
)
from .performance_cache import MetricType, PerformanceMonitor
from .prompt_manager import PromptManager

# Type alias for better type checking
PromptType = Any

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class EnhancedLLMService(Generic[T]):
    """
    Enhanced LLM service with intelligent model selection.

    Integrates the semantic model selection system with provider interfaces,
    enabling intelligent model selection based on task requirements, performance
    metrics, and fallback chains.
    """

    def __init__(
        self,
        config: LLMConfig,
        model_registry: Optional[ModelRegistry] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
    ):
        """Initialize the enhanced LLM service."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize provider factory and create providers
        self.provider_factory = get_provider_factory()
        self.providers = self.provider_factory.create_providers_from_config(config)

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize model selection components
        self.model_registry = model_registry or ModelRegistry()
        self.model_scorer = ModelScorer()
        # Create a dummy capability discovery for now
        from .capabilities.discovery import CapabilityDiscovery

        capability_discovery = CapabilityDiscovery(self.providers)
        self.model_selector = ModelSelector(
            self.model_registry, capability_discovery, self.model_scorer
        )
        self.performance_monitor = performance_monitor or PerformanceMonitor()

        # Populate model registry with models from config
        self._populate_model_registry()

        # Track selection statistics
        self._selection_stats: Dict[str, int] = {}
        self._last_selection_time: Dict[str, float] = {}

        self.logger.info(
            "EnhancedLLMService initialized with intelligent model selection"
        )

    def _populate_model_registry(self):
        """Populate the model registry with models from the LLM config."""
        from .model_registry import ModelCapability, ModelInfo

        for provider_name, provider_config in self.config.providers.items():
            # Convert string provider to ProviderType enum
            from .common.enums import ProviderType

            provider_type = ProviderType(provider_config.provider)
            for model_name, model_config in provider_config.models.items():
                # Convert capabilities from strings to ModelCapability enums
                capabilities = []
                for cap_str in model_config.capabilities:
                    try:
                        capabilities.append(
                            ModelCapability(
                                name=cap_str,
                                description=f"Capability: {cap_str}",
                                performance_score=getattr(
                                    model_config, "performance_score", 0.5
                                ),
                                cost_efficiency=1.0
                                - (
                                    model_config.cost_per_1k_tokens / 0.1
                                ),  # Normalize cost to efficiency
                            )
                        )
                    except (ValueError, AttributeError) as e:
                        # Skip unknown capabilities
                        self.logger.warning(f"Unknown capability {cap_str}: {e}")
                        continue

                # Create ModelInfo
                model_info = ModelInfo(
                    name=model_name,
                    provider=provider_type,
                    semantic_type=model_config.model_type,
                    capabilities=capabilities,
                    cost_per_1k_tokens=model_config.cost_per_1k_tokens,
                    max_tokens=model_config.max_tokens,
                    context_window=model_config.max_tokens,  # Use max_tokens as context window
                    performance_score=model_config.performance_score,
                    reliability_score=model_config.reliability_score,
                    provider_specific=provider_config.provider_specific,
                )

                # Register the model
                self.model_registry.register_model(model_info)
                self.logger.debug(
                    f"Registered model: {model_name} from provider: {provider_name}"
                )

        self.logger.info(
            f"Model registry populated with {self.model_registry.get_model_count()} models"
        )

    async def generate_structured(
        self,
        prompt: Union[str, Any],
        response_model: Type[T],
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider: Optional[str] = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE,
        custom_weights: Optional[ScoringWeights] = None,
        max_cost: Optional[float] = None,
        min_performance: Optional[float] = None,
        min_reliability: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Generate a structured response with intelligent model selection."""
        start_time = time.time()

        try:
            # Select the best model based on criteria
            selected_model, selection_result = await self._select_model_for_task(
                model=model,
                model_type=model_type,
                provider=provider,
                selection_strategy=selection_strategy,
                custom_weights=custom_weights,
                max_cost=max_cost,
                min_performance=min_performance,
                min_reliability=min_reliability,
                required_capabilities=[],  # Could be enhanced to detect from response_model
            )

            # Get the provider for the selected model
            provider_name = selected_model.provider.value
            if provider_name not in self.providers:
                raise ValueError(
                    f"Provider '{provider_name}' not available for model '{selected_model.name}'"
                )

            provider_instance = self.providers[provider_name]

            self.logger.info(
                f"Generating structured response using model: {selected_model.name} via provider: {provider_name}"
            )

            # Generate the response using regular generate method
            # Convert prompt to LLMRequest format with structured output instruction
            if isinstance(prompt, str):
                # Enhance the prompt to request structured JSON output
                # Check if this is for triage, analysis, or remediation based on response model
                if response_model.__name__ == "TriageResponse":
                    structured_prompt = f"""Please triage the following issue and provide a structured JSON response:

{prompt}

Please respond with a valid JSON object that includes all required fields for triage. The response should be in this format:
{{
    "severity": "low|medium|high|critical",
    "category": "Issue category (e.g., error, warning, performance)",
    "urgency": "low|medium|high|critical",
    "description": "Brief description of the issue",
    "suggested_actions": ["action1", "action2", "action3"]
}}

Respond only with the JSON object, no additional text."""
                elif response_model.__name__ == "RemediationResponse":
                    structured_prompt = f"""Please provide a detailed remediation plan for the following problem and respond with a structured JSON:

{prompt}

Please respond with a valid JSON object that includes all required fields for remediation. The response should be in this format:
{{
    "root_cause_analysis": "Detailed analysis of what caused the issue",
    "proposed_fix": "Description of the proposed solution",
    "code_patch": "Actual code changes needed (in Git patch format if applicable)",
    "priority": "low|medium|high|critical",
    "estimated_effort": "Estimated time/effort required (e.g., '2 hours', '1 day', 'immediate')"
}}

Focus on providing actionable, specific solutions with actual code when applicable. Respond only with the JSON object, no additional text."""
                else:
                    structured_prompt = f"""Please analyze the following and provide a structured JSON response:

{prompt}

Please respond with a valid JSON object that includes all required fields for the analysis. The response should be in this format:
{{
    "summary": "Brief summary of the analysis",
    "scores": {{"criterion1": 0.8, "criterion2": 0.6}},
    "key_points": ["point1", "point2", "point3"],
    "recommendations": ["recommendation1", "recommendation2"]
}}

Respond only with the JSON object, no additional text."""

                request = LLMRequest(
                    prompt=structured_prompt,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 1000),
                    model_type=model_type,
                )
            else:
                # Assume it's already a structured prompt
                request = prompt

            response = await provider_instance.generate(request)

            # Parse the response into the structured format
            try:
                import json
                import re

                # Extract JSON from response (in case there's extra text)
                json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_data = json.loads(json_str)
                    result = response_model(**parsed_data)
                else:
                    # Fallback: create a basic response with the raw content
                    if response_model.__name__ == "TriageResponse":
                        result = response_model(
                            severity="medium",
                            category="unknown",
                            urgency="medium",
                            description=(
                                response.content[:200] + "..."
                                if len(response.content) > 200
                                else response.content
                            ),
                            suggested_actions=["Investigate further"],
                        )
                    elif response_model.__name__ == "RemediationResponse":
                        result = response_model(
                            root_cause_analysis=(
                                response.content[:200] + "..."
                                if len(response.content) > 200
                                else response.content
                            ),
                            proposed_fix="Manual review required",
                            code_patch="# TODO: Generate proper code patch\n# "
                            + response.content[:100],
                            priority="medium",
                            estimated_effort="Unknown",
                        )
                    else:
                        result = response_model(
                            summary=(
                                response.content[:200] + "..."
                                if len(response.content) > 200
                                else response.content
                            ),
                            scores={"confidence": 0.5},
                            key_points=[
                                (
                                    response.content[:100] + "..."
                                    if len(response.content) > 100
                                    else response.content
                                )
                            ],
                            recommendations=[],
                        )
            except (json.JSONDecodeError, ValueError):  # type: ignore
                # Fallback: create a basic response with the raw content
                if response_model.__name__ == "TriageResponse":
                    result = response_model(
                        severity="medium",
                        category="unknown",
                        urgency="medium",
                        description=(
                            response.content[:200] + "..."
                            if len(response.content) > 200
                            else response.content
                        ),
                        suggested_actions=["Investigate further"],
                    )
                elif response_model.__name__ == "RemediationResponse":
                    result = response_model(
                        root_cause_analysis=(
                            response.content[:200] + "..."
                            if len(response.content) > 200
                            else response.content
                        ),
                        proposed_fix="Manual review required",
                        code_patch="# TODO: Generate proper code patch\n# "
                        + response.content[:100],
                        priority="medium",
                        estimated_effort="Unknown",
                    )
                else:
                    result = response_model(
                        summary=(
                            response.content[:200] + "..."
                            if len(response.content) > 200
                            else response.content
                        ),
                        scores={"confidence": 0.5},
                        key_points=[
                            (
                                response.content[:100] + "..."
                                if len(response.content) > 100
                                else response.content
                            )
                        ],
                        recommendations=[],
                    )

            # Record performance metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            self.performance_monitor.record_latency(
                model_name=selected_model.name,
                latency_ms=latency_ms,
                provider=selected_model.provider,
                context={
                    "task_type": "structured_generation",
                    "model_type": model_type.value if model_type else "unknown",
                    "selection_strategy": selection_strategy.value,
                    "response_model": response_model.__name__,
                },
            )

            self.performance_monitor.record_success(
                model_name=selected_model.name,
                success=True,
                provider=selected_model.provider,
                context={"task_type": "structured_generation"},
            )

            # Update selection statistics
            self._update_selection_stats(selected_model.name, selection_strategy)

            return result

        except Exception as e:
            # Record failure metrics if we have a selected model
            try:
                if "selected_model" in locals() and "selected_model" in locals():
                    selected_model = locals().get("selected_model")
                    if selected_model:
                        self.performance_monitor.record_success(
                            model_name=selected_model.name,
                            success=False,
                            provider=selected_model.provider,
                            context={
                                "task_type": "structured_generation",
                                "error": str(e),
                            },
                        )
            except Exception as metrics_error:
                self.logger.warning(f"Failed to record metrics: {metrics_error}")

            self.logger.error(f"Error generating structured response: {str(e)}")
            raise

    async def generate_text(
        self,
        prompt: Union[str, Any],
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider: Optional[str] = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE,
        custom_weights: Optional[ScoringWeights] = None,
        max_cost: Optional[float] = None,
        min_performance: Optional[float] = None,
        min_reliability: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a plain text response with intelligent model selection."""
        start_time = time.time()

        try:
            # Select the best model based on criteria
            selected_model, selection_result = await self._select_model_for_task(
                model=model,
                model_type=model_type,
                provider=provider,
                selection_strategy=selection_strategy,
                custom_weights=custom_weights,
                max_cost=max_cost,
                min_performance=min_performance,
                min_reliability=min_reliability,
                required_capabilities=[],
            )

            # Get the provider for the selected model
            provider_name = selected_model.provider.value
            if provider_name not in self.providers:
                raise ValueError(
                    f"Provider '{provider_name}' not available for model '{selected_model.name}'"
                )

            provider_instance = self.providers[provider_name]

            self.logger.info(
                f"Generating text response using model: {selected_model.name} via provider: {provider_name}"
            )

            # Generate the response using the correct interface
            request = LLMRequest(
                prompt=prompt, model_type=selected_model.semantic_type, **kwargs
            )
            response = await provider_instance.generate(request)
            result = response.content

            # Record performance metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            self.performance_monitor.record_latency(
                model_name=selected_model.name,
                latency_ms=latency_ms,
                provider=selected_model.provider,
                context={
                    "task_type": "text_generation",
                    "model_type": model_type.value if model_type else "unknown",
                    "selection_strategy": selection_strategy.value,
                },
            )

            self.performance_monitor.record_success(
                model_name=selected_model.name,
                success=True,
                provider=selected_model.provider,
                context={"task_type": "text_generation"},
            )

            # Update selection statistics
            self._update_selection_stats(selected_model.name, selection_strategy)

            return result

        except Exception as e:
            # Record failure metrics if we have a selected model
            try:
                if "selected_model" in locals():
                    selected_model = locals().get("selected_model")
                    if selected_model:
                        self.performance_monitor.record_success(
                            model_name=selected_model.name,
                            success=False,
                            provider=selected_model.provider,
                            context={"task_type": "text_generation", "error": str(e)},
                        )
            except Exception as metrics_error:
                self.logger.warning(f"Failed to record metrics: {metrics_error}")

            self.logger.error(f"Error generating text response: {str(e)}")
            raise

    async def generate_with_fallback(
        self,
        prompt: Union[str, Any],
        response_model: Optional[Type[T]] = None,
        model_type: Optional[ModelType] = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> Union[str, T]:
        """Generate response with automatic fallback chain execution."""
        last_error = None

        # Create selection criteria
        criteria = SelectionCriteria(
            semantic_type=model_type,
            strategy=selection_strategy,
            allow_fallback=True,
            max_models_to_consider=10,
        )

        # Get selection result with fallback chain
        selection_result = self.model_selector.select_model(criteria)

        # Try each model in the fallback chain
        for i, model_info in enumerate(selection_result.fallback_chain):
            if i >= max_attempts:
                break

            try:
                self.logger.info(
                    f"Attempting model: {model_info.name} (attempt {i+1}/{max_attempts})"
                )

                # Get the provider for this model
                provider_name = model_info.provider.value
                if provider_name not in self.providers:
                    self.logger.warning(
                        f"Provider '{provider_name}' not available for model '{model_info.name}'"
                    )
                    continue

                provider_instance = self.providers[provider_name]

                # Generate response using the correct interface
                request = LLMRequest(
                    prompt=prompt, model_type=model_info.semantic_type, **kwargs
                )
                response = await provider_instance.generate(request)
                result = response.content

                # Record success
                self.performance_monitor.record_success(
                    model_name=model_info.name,
                    success=True,
                    provider=model_info.provider,
                    context={"task_type": "fallback_generation", "attempt": i + 1},
                )

                self.logger.info(
                    f"Successfully generated response using model: {model_info.name}"
                )
                return result

            except Exception as e:
                last_error = e
                self.logger.warning(f"Model {model_info.name} failed: {str(e)}")

                # Record failure
                self.performance_monitor.record_success(
                    model_name=model_info.name,
                    success=False,
                    provider=model_info.provider,
                    context={
                        "task_type": "fallback_generation",
                        "attempt": i + 1,
                        "error": str(e),
                    },
                )

                continue

        # All models failed
        self.logger.error(
            f"All {max_attempts} models in fallback chain failed. Last error: {str(last_error)}"
        )
        raise last_error or Exception("All models in fallback chain failed")

    async def _select_model_for_task(
        self,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider: Optional[str] = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.BEST_SCORE,
        custom_weights: Optional[ScoringWeights] = None,
        max_cost: Optional[float] = None,
        min_performance: Optional[float] = None,
        min_reliability: Optional[float] = None,
        required_capabilities: Optional[List] = None,
    ) -> tuple[ModelInfo, SelectionResult]:
        """Select the best model for a task based on criteria."""

        # If specific model is requested, try to find it in registry
        if model:
            model_info = self.model_registry.get_model(model)
            if model_info:
                # Create a simple selection result
                context = ScoringContext(
                    task_type=model_type,
                    required_capabilities=required_capabilities or [],
                    max_cost=max_cost,
                    min_performance=min_performance,
                    min_reliability=min_reliability,
                )
                score = self.model_scorer.score_model(model_info, context)

                selection_result = SelectionResult(
                    selected_model=model_info,
                    score=score,
                    fallback_chain=[model_info],
                    selection_reason=f"Explicitly requested model: {model}",
                    criteria=SelectionCriteria(
                        semantic_type=model_type,
                        strategy=selection_strategy,
                        custom_weights=custom_weights,
                        max_cost=max_cost,
                        min_performance=min_performance,
                        min_reliability=min_reliability,
                    ),
                )
                return model_info, selection_result

        # Create selection criteria
        criteria = SelectionCriteria(
            semantic_type=model_type,
            required_capabilities=required_capabilities or [],
            max_cost=max_cost,
            min_performance=min_performance,
            min_reliability=min_reliability,
            provider_preference=ProviderType(provider) if provider else None,
            strategy=selection_strategy,
            custom_weights=custom_weights,
            allow_fallback=True,
        )

        # Select model with fallback support
        selected_model, selection_result = (
            self.model_selector.select_model_with_fallback(criteria)
        )

        return selected_model, selection_result

    def _update_selection_stats(
        self, model_name: str, strategy: SelectionStrategy
    ) -> None:
        """Update selection statistics."""
        key = f"{model_name}:{strategy.value}"
        self._selection_stats[key] = self._selection_stats.get(key, 0) + 1
        self._last_selection_time[model_name] = time.time()

    async def health_check(self, provider: Optional[str] = None) -> bool:
        """Check if the specified provider is healthy and accessible."""
        try:
            if provider:
                if provider in self.providers:
                    return await self.providers[provider].health_check()
                return False

            # Check all providers
            health_status = {}
            for provider_name, provider_instance in self.providers.items():
                health_status[provider_name] = await provider_instance.health_check()

            return all(health_status.values())

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    def get_available_models(
        self, provider: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get available models for the specified provider or all providers."""
        if provider:
            if provider in self.providers:
                models = self.providers[provider].get_available_models()
                return {
                    provider: (
                        list(models.values()) if isinstance(models, dict) else models
                    )
                }
            return {}

        result = {}
        for provider_name, provider_instance in self.providers.items():
            models = provider_instance.get_available_models()
            result[provider_name] = (
                list(models.values()) if isinstance(models, dict) else models
            )
        return result

    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        return self.performance_monitor.get_model_performance(model_name)

    def get_best_models(
        self, metric_type: MetricType, limit: int = 5
    ) -> List[tuple[str, float]]:
        """Get best performing models for a specific metric."""
        return self.performance_monitor.get_best_models(metric_type, limit)

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get model selection statistics."""
        return {
            "selection_counts": self._selection_stats.copy(),
            "last_selection_times": self._last_selection_time.copy(),
            "performance_cache_stats": self.performance_monitor.get_cache_stats(),
        }

    def get_model_rankings(
        self,
        metric_types: List[MetricType],
        weights: Optional[Dict[MetricType, float]] = None,
    ) -> List[tuple[str, float]]:
        """Get model rankings based on multiple performance metrics."""
        return self.performance_monitor.get_model_rankings(metric_types, weights)


def create_enhanced_llm_service(
    config: LLMConfig,
    model_registry: Optional[ModelRegistry] = None,
    performance_monitor: Optional[PerformanceMonitor] = None,
) -> EnhancedLLMService:
    """Factory function to create and configure an EnhancedLLMService instance."""
    return EnhancedLLMService(config, model_registry, performance_monitor)
