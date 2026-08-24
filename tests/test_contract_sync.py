"""The three contract files must agree, checked rather than remembered.

`CLAUDE.md` says the wire contract lives in three files that must stay in sync:

    packages/shared_contracts/schemas.py     (Pydantic, the source of truth)
    packages/shared_contracts/types.ts       (shared TS mirror)
    apps/frontend/src/types/diarization.ts   (frontend working copy)

Nothing enforced that. Keeping three hand-maintained copies aligned by discipline
is exactly the kind of rule that holds until the one time it doesn't, and the
failure is silent: the backend sends a field the UI never reads, or the UI reads a
field that is always undefined. Both look like a UI bug.

This parses the TypeScript rather than importing it — a regex over interface
bodies is enough to compare FIELD NAMES, which is where drift actually happens.
It deliberately does not try to compare types; that would need a real TS parser
and would fail on every legitimate representation difference (Python `None` vs TS
`?`, `dict` vs `Record`).
"""

import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from packages.shared_contracts import schemas

REPO_ROOT = Path(__file__).resolve().parents[1]
TS_FILES = (
    REPO_ROOT / "packages/shared_contracts/types.ts",
    REPO_ROOT / "apps/frontend/src/types/diarization.ts",
)

#: Contracts that cross the wire and therefore must exist in all three files.
#: Every ContractModel subclass is discovered automatically, so a new one is
#: covered the moment it is defined -- there is no list here to forget to update.
CONTRACT_MODELS = sorted(
    name
    for name, obj in vars(schemas).items()
    if isinstance(obj, type)
    and issubclass(obj, BaseModel)
    and obj is not schemas.ContractModel
    and issubclass(obj, schemas.ContractModel)
)


def _ts_interfaces(source: str) -> dict[str, set[str]]:
    """Field names per `export interface`, comments and nesting stripped."""
    interfaces: dict[str, set[str]] = {}
    for match in re.finditer(r"export interface (\w+)\s*\{", source):
        name = match.group(1)
        depth, index = 1, match.end()
        while depth and index < len(source):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        body = source[match.end():index - 1]
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        body = re.sub(r"//.*", "", body)
        # Top-level `name?: type;` only -- a nested object literal's own keys are
        # not fields of this interface.
        fields = set()
        nesting = 0
        for line in body.splitlines():
            stripped = line.strip()
            if nesting == 0:
                field = re.match(r"(\w+)\??\s*:", stripped)
                if field:
                    fields.add(field.group(1))
            nesting += stripped.count("{") - stripped.count("}")
        interfaces[name] = fields
    return interfaces


def _ts_type_aliases(source: str) -> set[str]:
    return set(re.findall(r"export type (\w+)\s*=", source))


@pytest.fixture(scope="module")
def parsed() -> dict[Path, tuple[dict[str, set[str]], set[str]]]:
    return {path: (_ts_interfaces(path.read_text()), _ts_type_aliases(path.read_text()))
            for path in TS_FILES}


def test_contract_models_were_discovered() -> None:
    """Guard the discovery itself: if this list empties out, every test below
    would pass while checking nothing."""
    assert "TranscriptRun" in CONTRACT_MODELS
    assert "DiarizationModelRun" in CONTRACT_MODELS
    assert len(CONTRACT_MODELS) >= 10


@pytest.mark.parametrize("model_name", CONTRACT_MODELS)
@pytest.mark.parametrize("ts_path", TS_FILES, ids=lambda p: p.name)
def test_every_contract_has_a_matching_typescript_interface(model_name, ts_path, parsed) -> None:
    interfaces, _ = parsed[ts_path]
    assert model_name in interfaces, (
        f"{model_name} exists in schemas.py but not in {ts_path.name} — "
        "the wire contract lives in three files and they must agree"
    )


@pytest.mark.parametrize("model_name", CONTRACT_MODELS)
@pytest.mark.parametrize("ts_path", TS_FILES, ids=lambda p: p.name)
def test_field_names_match_the_typescript_interface(model_name, ts_path, parsed) -> None:
    interfaces, _ = parsed[ts_path]
    if model_name not in interfaces:
        pytest.skip("covered by the interface-presence test")

    model = getattr(schemas, model_name)
    # by_alias: the wire format is camelCase, which is what the TS mirrors.
    expected = {field.alias or name for name, field in model.model_fields.items()}
    actual = interfaces[model_name]

    assert expected == actual, (
        f"{model_name} field mismatch with {ts_path.name}\n"
        f"  only in schemas.py : {sorted(expected - actual)}\n"
        f"  only in {ts_path.name}: {sorted(actual - expected)}"
    )


@pytest.mark.parametrize(
    "alias_name",
    ["ModelStatus", "TranscriptStage", "TranscriptionMode", "TranscriptSource",
     "TranscriptTransport", "ReferenceSource", "AlignmentOp", "ModelLifecycleState",
     "TtsDelivery"],
)
@pytest.mark.parametrize("ts_path", TS_FILES, ids=lambda p: p.name)
def test_literal_unions_exist_in_typescript(alias_name, ts_path, parsed) -> None:
    _, aliases = parsed[ts_path]
    assert alias_name in aliases, f"{alias_name} missing from {ts_path.name}"


def test_the_two_typescript_files_are_identical_in_contract_surface(parsed) -> None:
    """The frontend copy is a working copy, not a fork. Same interfaces, same
    fields — if they drift, one of them is lying about the wire."""
    shared, frontend = (parsed[path][0] for path in TS_FILES)
    assert set(shared) == set(frontend), (
        f"interfaces differ:\n"
        f"  only in types.ts       : {sorted(set(shared) - set(frontend))}\n"
        f"  only in diarization.ts : {sorted(set(frontend) - set(shared))}"
    )
    for name in sorted(set(shared) & set(frontend)):
        assert shared[name] == frontend[name], f"{name} fields differ between the two TS files"
