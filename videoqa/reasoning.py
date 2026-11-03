"""Reasoning stage using the single prompt and model reported in the paper."""

import os
from abc import ABC, abstractmethod

from .common_types import FusedContent
from .config import ReasoningConfig

PROMPT_TEMPLATE = """You are given a structured description of a video that contains visual
observations along with audio or speech content that captures the
progression of scenes or events over time. Each description segment is
separated using the pipe symbol (|). Based on the following video
content description, please answer the question:

VIDEO CONTENT:
{context}

QUESTION: {question}

ANSWER:"""


def _load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def select_context(fused_content: FusedContent, context_source: str) -> str:
    """Select one of the three paper-defined context compositions."""
    if context_source == "captions_only":
        return fused_content.captions_text
    if context_source == "transcripts_only":
        return fused_content.transcripts_text
    if context_source == "multimodal":
        return fused_content.timeline_text
    raise ValueError(f"Unsupported context source: {context_source}")


def build_prompt(fused_content: FusedContent, question: str, context_source: str) -> str:
    return PROMPT_TEMPLATE.format(
        context=select_context(fused_content, context_source),
        question=question,
    )


class LLMReasoner(ABC):
    @abstractmethod
    def answer_question(self, fused_content: FusedContent, question: str) -> str:
        raise NotImplementedError


class OpenAIReasoner(LLMReasoner):
    """Call GPT-4o-mini once at temperature zero for each query."""

    def __init__(self, config: ReasoningConfig):
        self.config = config
        _load_env_if_available()
        from openai import OpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the reasoning stage")
        self.client = OpenAI(api_key=api_key)

    def answer_question(self, fused_content: FusedContent, question: str) -> str:
        prompt = build_prompt(fused_content, question, self.config.context_source)
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


def get_reasoner(config: ReasoningConfig) -> LLMReasoner:
    if config.provider != "openai":
        raise ValueError("The paper method uses only the OpenAI reasoning provider")
    return OpenAIReasoner(config)
