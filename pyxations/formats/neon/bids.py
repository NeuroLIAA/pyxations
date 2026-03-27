import json
import shutil
import pandas as pd
from pathlib import Path


class NeonBidsConverter:
    """
    BIDS converter for Pupil Labs Neon recordings.

    A Neon recording is a folder (e.g. '2026-03-27-15-12-46/') containing
    info.json, gaze ps1.raw, etc.  The subject/session names are read from
    wearer.json ('name' field, expected format: 'sub-<id>_ses-<id>').
    """

    def relevant_extensions(self):
        # We use info.json as the representative file for each recording folder.
        return [".json"]

    def get_subject_ids(self, file_paths):
        ids = set()
        for p in file_paths:
            if p.name == "info.json":
                subj = self._subject_id_from_recording(p.parent)
                if subj:
                    ids.add(subj)
        return list(ids)

    def move_file_to_bids_folder(self, file, bids_folder_path, subject_id, old_subject_id, session_id):
        if file.name != "info.json":
            return
        recording_folder = file.parent
        subj, ses = self._parse_wearer_name(recording_folder)
        if subj != old_subject_id:
            return
        dest = Path(bids_folder_path) / f"sub-{subject_id}" / f"ses-{ses}" / "ET" / recording_folder.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copytree(recording_folder, dest)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_wearer_name(self, recording_folder: Path):
        """Return (subject_id, session_id) from wearer.json name field."""
        wearer_file = recording_folder / "wearer.json"
        if not wearer_file.exists():
            return None, None
        name = json.loads(wearer_file.read_text()).get("name", "")
        # Expected format: 'sub-01_ses-1'
        parts = {p.split("-")[0]: p.split("-", 1)[1] for p in name.split("_") if "-" in p}
        return parts.get("sub"), parts.get("ses")

    def _subject_id_from_recording(self, recording_folder: Path):
        subj, _ = self._parse_wearer_name(recording_folder)
        return subj


def recording_to_bids(recording_folder: Path, dataset_root: Path) -> Path:
    """
    Organize a single Neon recording into a BIDS dataset folder and return
    its path.  Creates participants.tsv and the sub-/ses-/ET/ structure.

    Args:
        recording_folder: Path to the Neon recording folder (contains info.json).
        dataset_root:     Where to create the BIDS dataset.

    Returns:
        Path to the BIDS dataset root (dataset_root / 'neon_dataset').
    """
    recording_folder = Path(recording_folder)
    dataset_root = Path(dataset_root)

    converter = NeonBidsConverter()
    subj_id, ses_id = converter._parse_wearer_name(recording_folder)
    if not subj_id or not ses_id:
        raise ValueError(
            f"Cannot parse subject/session from wearer.json in {recording_folder}. "
            "Expected name format: 'sub-<id>_ses-<id>'."
        )

    new_subj_id = subj_id.zfill(4)
    bids_folder = dataset_root
    bids_folder.mkdir(parents=True, exist_ok=True)

    # participants.tsv
    tsv = bids_folder / "participants.tsv"
    if not tsv.exists():
        pd.DataFrame([{"subject_id": new_subj_id, "old_subject_id": subj_id}]).to_csv(
            tsv, sep="\t", index=False
        )

    # Copy recording folder into BIDS structure
    dest = bids_folder / f"sub-{new_subj_id}" / f"ses-{ses_id}" / "ET" / recording_folder.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copytree(recording_folder, dest)

    return bids_folder
