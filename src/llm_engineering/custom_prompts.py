from __future__ import annotations

from src.utils.placeholders import not_implemented_placeholder


def load_custom_rule_prompt_bundle(*args, **kwargs):
    not_implemented_placeholder(
        "load_custom_rule_prompt_bundle",
        "Provide the custom prompt assets for LLM-engineered rule generation.",
    )


def render_custom_rule_prompt(*args, **kwargs):
    not_implemented_placeholder(
        "render_custom_rule_prompt",
        "Implement custom prompt rendering before running prompt-driven rule generation.",
    )


def postprocess_generated_rules(*args, **kwargs):
    not_implemented_placeholder(
        "postprocess_generated_rules",
        "Implement project-specific rule post-processing before using generated rule families.",
    )


def generate_custom_engineered_rule_family(*args, **kwargs):
    not_implemented_placeholder(
        "generate_custom_engineered_rule_family",
        "Custom prompt-driven engineered rule generation has not been implemented yet.",
    )
