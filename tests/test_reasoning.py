from videoqa.common_types import FusedContent
from videoqa.reasoning import PROMPT_TEMPLATE, build_prompt, select_context


def fused():
    return FusedContent(
        timeline_text="both",
        narrative_text="both",
        captions_text="visual",
        transcripts_text="speech",
    )


def test_context_ablations_select_only_the_reported_channel():
    assert select_context(fused(), "multimodal") == "both"
    assert select_context(fused(), "captions_only") == "visual"
    assert select_context(fused(), "transcripts_only") == "speech"


def test_prompt_is_the_shared_paper_template():
    prompt = build_prompt(fused(), "What happened?", "multimodal")
    assert prompt == PROMPT_TEMPLATE.format(context="both", question="What happened?")
    assert "pipe symbol (|)" in prompt
