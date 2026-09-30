from pathlib import Path

PROMPTS = Path("src/swaag/assets/prompts")


def test_agent_action_contract_makes_audio_style_the_default_user_facing_format():
    text = (PROMPTS / "agent_action_user.txt").read_text()
    assert "assistant_message must use audio style" in text
    assert "continuous spoken prose" in text
    assert "spoken/rounded numbers" in text
    assert "machine-readable formatting" in text


def test_status_answer_is_audio_style_but_structured_fields_are_exempt():
    text = (PROMPTS / "communication_status_system.txt").read_text()
    assert "user-facing `answer` field" in text
    assert "must use audio style by default" in text
    assert "Other JSON fields are machine data and exempt" in text


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
