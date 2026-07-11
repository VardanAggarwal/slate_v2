"""Edit = supersede. Episodes are immutable (PLAN.md §10 — enforced by trigger),
so editing a note saves a NEW episode carrying the revised text and masks the old
one behind an EPISODE_SUPERSEDED event. The applier (consolidate.apply_event)
clears the old episode's derived artifacts — fragments, claim support, claims
only it supported, its FTS row — and read paths exclude superseded episodes; the
revised episode then re-derives through the normal refine → consolidate path.

The revised episode KEEPS the original timestamp: an edit revises that thought,
it doesn't restate it today (retrieval time signals and chronology stay stable).
The edit time itself lives on the event / supersession row.

Ordering: encode the new episode first (its own transaction) — if it fails, the
old note is untouched. The supersession transaction follows; its failure window
is a single local write. mark_fragmented claims the OLD episode's fragmentation
marker inside that transaction, so an in-flight async refine of the old note
loses the race and skips — no stale fragments can land after the applier's
delete.
"""
from core import store
from core.encode import encode


def edit_note(conn, user_id: str, episode_id: str, new_text: str,
              title: str | None = None) -> dict:
    """Supersede one note with revised text. Returns the new episode's receipt
    plus `superseded_episode_id` / `title`. Raises ValueError on unknown or
    already-superseded notes."""
    old = store.get_episode(conn, user_id, episode_id)
    if not old:
        raise ValueError(f"Note not found: {episode_id}")
    current = store.episode_superseded_by(conn, user_id, episode_id)
    if current:
        raise ValueError(f"{episode_id} was already edited — the current version "
                         f"is {current}; edit that one instead.")

    resolved_title = (title or old["title"] or "").strip() or None
    receipt = encode(conn, user_id, new_text, ts=old["ts"], title=resolved_title,
                     source="edit", exclude_episode_ids={episode_id})

    payload = {"old_id": episode_id, "new_id": receipt["episode_id"],
               "ts": store.now_iso()}
    from core.consolidate import apply_event
    with conn:
        store.mark_fragmented(conn, user_id, episode_id, 0)
        store.append_event(conn, user_id, "EPISODE_SUPERSEDED", payload)
        apply_event(conn, user_id, "EPISODE_SUPERSEDED", payload)

    receipt["superseded_episode_id"] = episode_id
    receipt["title"] = resolved_title
    return receipt
