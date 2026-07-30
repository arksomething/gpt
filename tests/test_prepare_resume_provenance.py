"""Gate 0: --resume must refuse to reuse .bin files whose tokenizer
provenance is missing or mismatched (the v3 val.bin incident)."""

import json
import os

import pytest

from scripts.prepare_data import Checkpoint, verify_flat_resume_provenance

SHA_A = "a" * 64
SHA_B = "b" * 64


def _make_bin(tmp_path, name="val.bin", n_bytes=1000):
    path = tmp_path / name
    path.write_bytes(b"\x00" * n_bytes)
    return str(path)


def _make_meta(tmp_path, tokenizer_sha256):
    meta = tmp_path / "data_meta.json"
    meta.write_text(json.dumps({"tokenizer_sha256": tokenizer_sha256}))
    return str(meta)


def _checkpoint(sha):
    return Checkpoint(
        phase="val",
        tokens_written=1,
        docs_processed=1,
        c4_docs=0,
        wiki_docs=0,
        fineweb_docs=0,
        dedup_rejects=0,
        elapsed_seconds=0.0,
        timestamp="t",
        tokenizer_sha256=sha,
    )


def test_no_existing_bins_is_fine(tmp_path):
    verify_flat_resume_provenance(
        str(tmp_path), [str(tmp_path / "val.bin")], SHA_A, None
    )


def test_matching_meta_is_fine(tmp_path):
    bin_path = _make_bin(tmp_path)
    _make_meta(tmp_path, SHA_A)
    verify_flat_resume_provenance(str(tmp_path), [bin_path], SHA_A, None)


def test_mismatched_meta_aborts(tmp_path):
    bin_path = _make_bin(tmp_path)
    _make_meta(tmp_path, SHA_B)
    with pytest.raises(SystemExit, match="Tokenizer mismatch"):
        verify_flat_resume_provenance(str(tmp_path), [bin_path], SHA_A, None)


def test_bins_without_any_provenance_abort(tmp_path):
    bin_path = _make_bin(tmp_path)
    with pytest.raises(SystemExit, match="no\ntokenizer provenance|no "):
        verify_flat_resume_provenance(str(tmp_path), [bin_path], SHA_A, None)


def test_checkpoint_provenance_accepted_when_meta_absent(tmp_path):
    bin_path = _make_bin(tmp_path)
    verify_flat_resume_provenance(
        str(tmp_path), [bin_path], SHA_A, _checkpoint(SHA_A)
    )


def test_checkpoint_provenance_mismatch_aborts(tmp_path):
    bin_path = _make_bin(tmp_path)
    with pytest.raises(SystemExit, match="Tokenizer mismatch"):
        verify_flat_resume_provenance(
            str(tmp_path), [bin_path], SHA_A, _checkpoint(SHA_B)
        )


def test_checkpoint_roundtrips_tokenizer_sha():
    cp = _checkpoint(SHA_A)
    assert Checkpoint.from_dict(cp.to_dict()).tokenizer_sha256 == SHA_A


def test_legacy_checkpoint_without_sha_loads():
    d = _checkpoint(SHA_A).to_dict()
    del d["tokenizer_sha256"]
    assert Checkpoint.from_dict(d).tokenizer_sha256 is None
