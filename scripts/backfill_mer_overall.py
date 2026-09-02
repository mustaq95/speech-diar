"""One-shot: fill `mer`, `mer_raw`, `overall`, `overall_raw` on rows that
predate those columns.

Runs against the real Postgres pointed at by `.env`. The API already serves these
figures on the fly (evaluations.py falls back to the same identities when a stored
value is null), so scoreboard reads are already correct without this script — the
purpose here is to make the STORED columns match, so aggregate reports and any
future consumer see one consistent number for every row.

The MER identity is `(S+D+I) / (ref_words + I)`, exactly what `packages.metrics.wer`
computes for freshly scored rows. Overall is a straight mean of WER (capped at 1.0),
CER, and MER — same as the scoring path. No re-scoring is performed: the counts on
disk are authoritative, and re-scoring would recompute them and possibly disagree
by an amount that reads as a real change when it is a coincidence of retokenization.

Idempotent: rows whose mer/overall columns are already populated are left alone.
Report at the end so a rerun tells you "0 to backfill" instead of doing invisible
work.

    uv run python scripts/backfill_mer_overall.py            # run against the .env DB
    uv run python scripts/backfill_mer_overall.py --dry-run  # print what WOULD change
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.database.models import TranscriptResult  # noqa: E402
from packages.database.session import SessionLocal, init_db  # noqa: E402

logger = logging.getLogger("backfill_mer_overall")


def _mer(sub: int | None, dele: int | None, ins: int | None, ref_words: int | None) -> float:
    sub_v = sub or 0
    del_v = dele or 0
    ins_v = ins or 0
    ref_v = ref_words or 0
    errors = sub_v + del_v + ins_v
    denom = ref_v + ins_v
    return errors / denom if denom else 0.0


def _overall(wer: float | None, cer: float | None, mer: float) -> float:
    """Mean of WER, CER (each capped at 1.0), MER. Nulls treated as 0.0 to match
    how the scoring path treats a fresh row with an empty hypothesis: the
    alignment step returns zero counts and the derived rates come out zero."""
    return (min(wer or 0.0, 1.0) + min(cer or 0.0, 1.0) + mer) / 3.0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rows that would be updated; do not commit",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Same entry point the API and the supervisor use: idempotent, and it fires
    # the _ADDED_COLUMNS shim so mer / mer_raw / overall / overall_raw exist on
    # a DB that predates them. Without this, the SELECT below fails with
    # UndefinedColumn.
    init_db()

    updated = 0
    skipped = 0
    with SessionLocal() as session:
        # Only rows that actually have counts to derive from. Rows without a
        # reference never had a WER either, and we do not invent one here.
        rows = (
            session.query(TranscriptResult)
            .filter(TranscriptResult.ref_word_count.isnot(None))
            .all()
        )
        for row in rows:
            # Compute both normalized and raw. The raw counts are not stored
            # separately, so the raw MER falls back to the normalized MER on
            # legacy rows (same behaviour as the API's TranscriptMetrics fallback).
            mer_norm = _mer(row.sub_count, row.del_count, row.ins_count, row.ref_word_count)
            mer_raw = mer_norm  # per-row raw counts are not on the model
            overall_norm = _overall(row.wer, row.cer, mer_norm)
            overall_raw = _overall(row.wer_raw, row.cer_raw, mer_raw)

            changed = False
            if row.mer is None:
                row.mer = mer_norm
                changed = True
            if row.mer_raw is None:
                row.mer_raw = mer_raw
                changed = True
            if row.overall is None:
                row.overall = overall_norm
                changed = True
            if row.overall_raw is None:
                row.overall_raw = overall_raw
                changed = True

            if changed:
                updated += 1
                logger.info(
                    "audio_file_id=%s asr_id=%s source=%s -> WER=%.3f CER=%.3f MER=%.3f overall=%.3f",
                    row.audio_file_id,
                    row.asr_id,
                    row.source,
                    row.wer or 0.0,
                    row.cer or 0.0,
                    mer_norm,
                    overall_norm,
                )
            else:
                skipped += 1

        if args.dry_run:
            session.rollback()
            logger.info("DRY RUN — rolled back. %d row(s) would be updated, %d already had values.", updated, skipped)
        else:
            session.commit()
            logger.info("Backfilled %d row(s); %d already had values.", updated, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
