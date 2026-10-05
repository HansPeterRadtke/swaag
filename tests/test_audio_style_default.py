from pathlib import Path

PROMPTS = Path("src/swaag/assets/prompts")


def test_agent_action_contract_makes_audio_style_the_default_user_facing_format():
    text = (PROMPTS / "agent_action_user.txt").read_text()
    assert "assistant_message must use audio style" in text
    assert "continuous spoken prose" in text
    assert "spoken or appropriately rounded numbers" in text
    assert "machine-readable formatting" in text
    assert "Do not add apologies, reassurance, praise" in text
    assert "commentary on the user's anger" in text
    assert "Save the user's time" in text
    assert "code blocks, unnecessary line breaks" in text


def test_status_answer_is_audio_style_but_structured_fields_are_exempt():
    text = (PROMPTS / "communication_status_system.txt").read_text()
    assert "user-facing `answer` field" in text
    assert "must use audio style by default" in text
    assert "Other JSON fields are machine data and exempt" in text
    assert "Do not add apologies, reassurance, praise" in text
    assert "commentary on the user's anger" in text
    assert "Save the user's time" in text
    assert "code blocks, unnecessary line breaks" in text


def test_programming_prompt_requires_explicit_algorithm_selection_evidence():
    for name in ("system_standard.txt", "system_lean.txt"):
        text = (PROMPTS / name).read_text()
        assert "For nontrivial algorithms, define the objective" in text
        assert "candidate quality, time, memory, and failures" in text
    coding = (Path("src/swaag/tools/builtin.py").read_text() + Path("src/swaag/tools/terminal.py").read_text())
    assert "objective/error and constraints" in coding
    assert "representative and adversarial inputs" in coding
    assert "convergence/failure behavior" in coding
    assert "parameter/initialization sensitivity" in coding


def test_orchestrator_fast_interaction_uses_the_same_audio_style_default_and_exact_format_exception():
    text = Path("src/swaag/runtime.py").read_text()
    start = text.index("def generate_orchestrator_interaction")
    end = text.index("def generate_communication_status", start)
    prompt = text[start:end]
    assert "user-facing chat response" in prompt
    assert "must use audio style by default" in prompt
    assert "continuous spoken prose" in prompt
    assert "spoken or " in prompt
    assert "rounded numbers by default" in prompt
    assert "Do not add apologies, reassurance, praise" in prompt
    assert "commentary on the user's anger" in prompt
    assert "save the user's time" in prompt
    assert "exact, visual, " in prompt
    assert "preserve the requested exact content and format" in prompt


def test_optional_audio_renderer_does_not_reintroduce_user_facing_filler():
    text = (PROMPTS / "audio_rendering_system.txt").read_text()
    assert "Do not add apologies, reassurance, praise" in text
    assert "commentary on the user's anger" in text
    assert "implementation narration" in text
    assert "machine-noise identifiers" in text
