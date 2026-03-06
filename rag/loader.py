"""
loader.py — Load Indian law JSON files and normalize into uniform document dicts.

Each JSON file is an array of objects with schema:
    { "law_type": str, "number": str, "title": str, "text": str }

The loader reads every file in the data directory and returns a flat list of
standardized document dictionaries ready for chunking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class LegalDocument(TypedDict):
    """Standardized legal document dict."""
    act: str
    section: str
    title: str
    text: str


# ── Mapping from raw `law_type` values to clean act names ──────────────────
_ACT_NAME_MAP: dict[str, str] = {
    "IPC": "Indian Penal Code (IPC)",
    "CrPC": "Code of Criminal Procedure (CrPC)",
    "Constitution": "Constitution of India",
    "Domestic Violence Act": "Protection of Women from Domestic Violence Act",
    "Dowry Prohibition Act": "Dowry Prohibition Act",
    "Hindu Marriage Act": "Hindu Marriage Act",
    "Special Marriage Act": "Special Marriage Act",
}


def _normalize_act_name(raw: str) -> str:
    """Return a clean act name; fall back to the raw value if unknown."""
    return _ACT_NAME_MAP.get(raw, raw)


def load_single_file(filepath: Path) -> list[LegalDocument]:
    """Load a single JSON law file and return a list of LegalDocument dicts."""
    logger.info("Loading %s", filepath.name)
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            raw_records: list[dict] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load %s: %s", filepath, exc)
        return []

    documents: list[LegalDocument] = []
    for record in raw_records:
        act = _normalize_act_name(record.get("law_type", filepath.stem))
        section_num = str(record.get("number", "")).strip()
        title = record.get("title", "").strip()
        text = record.get("text", "").strip()

        if not text:
            logger.warning("Skipping empty text in %s Section %s", act, section_num)
            continue

        documents.append(
            LegalDocument(
                act=act,
                section=f"Section {section_num}" if section_num else "N/A",
                title=title,
                text=text,
            )
        )

    logger.info("  → loaded %d sections from %s", len(documents), filepath.name)
    return documents


def load_all_files(data_dir: str | Path) -> list[LegalDocument]:
    """
    Load every *.json file in *data_dir* and return a flat list of documents.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing the law JSON files.

    Returns
    -------
    list[LegalDocument]
        All legal documents across every file.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_path}")

    logger.info("Found %d JSON files in %s", len(json_files), data_path)

    all_docs: list[LegalDocument] = []
    for fp in json_files:
        all_docs.extend(load_single_file(fp))

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs
