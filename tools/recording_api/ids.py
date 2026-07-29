"""Deterministic (sessionId, agendaItemId) derivation from a filename.

UUIDv5 over a fixed namespace, so re-seeding the same sample files always
produces the same URLs — stable curls you can bookmark. UUIDs are already in
hyphenated (kebab) form, matching production ids like
`8a04c015-3345-4011-a2cd-c1f645ce78f7`.
"""

import uuid

NS = uuid.UUID("6f4d2e2a-1c3b-4f5a-9e7d-2b8a1c0d3e4f")


def ids_for(filename: str) -> tuple[str, str]:
    session_id = str(uuid.uuid5(NS, f"session:{filename}"))
    agenda_item_id = str(uuid.uuid5(NS, f"agenda:{filename}"))
    return session_id, agenda_item_id
