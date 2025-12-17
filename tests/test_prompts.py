"""Snapshot tests for llmwalk prompts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension

from llmwalk.cli import main

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class JSONSnapshotExtensionPretty(JSONSnapshotExtension):
    """JSON snapshot extension with pretty printing."""

    def serialize(self, data, **kwargs):
        return json.dumps(data, indent=2, sort_keys=True)


@pytest.fixture
def snapshot_json(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(JSONSnapshotExtensionPretty)


def get_prompt_files() -> list[Path]:
    """Get all prompt files from the prompts directory."""
    return sorted(PROMPTS_DIR.glob("*.txt"))


@pytest.mark.parametrize(
    "prompt_file",
    get_prompt_files(),
    ids=[p.stem for p in get_prompt_files()],
)
def test_prompt_snapshot(
    prompt_file: Path,
    snapshot_json: SnapshotAssertion,
    capsys,
) -> None:
    """Test that running llmwalk on each prompt produces consistent output."""
    # Run with JSON format for deterministic output
    main(["-p", str(prompt_file), "--format", "json"])

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    # Snapshot the structured output
    assert output == snapshot_json
