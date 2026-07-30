#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import sentencepiece as spm
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, hf_hub_download, login as hf_login

from scripts.indexed_shards import (
    IndexedCorpusSampler,
    IndexedShardReader,
    ShardFormatError,
)
from scripts.run_observer import RunObserver, atomic_write_json


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _find_artifact_manifest_path(checkpoint_dir):
    parent = os.path.dirname(os.path.normpath(checkpoint_dir))
    candidates = [
        os.path.join(checkpoint_dir, "artifacts_manifest.json"),
        os.path.join(checkpoint_dir, "artifacts", "artifacts_manifest.json"),
        os.path.join(parent, "artifacts_manifest.json"),
        os.path.join(parent, "artifacts", "artifacts_manifest.json"),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def _load_initial_weights(
    model,
    checkpoint_dir,
    *,
    model_config_path,
    tokenizer_path,
):
    """Load a base checkpoint for a new run and return immutable lineage."""

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        raise SystemExit(
            f"Initialization checkpoint has no model.pt: {checkpoint_dir}"
        )
    manifest_path = _find_artifact_manifest_path(checkpoint_dir)
    if manifest_path is None:
        raise SystemExit(
            "Initialization checkpoint is missing artifacts_manifest.json; "
            "refusing an untraceable post-training run."
        )
    manifest = _load_json_if_exists(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Invalid initialization manifest: {manifest_path}")

    errors = []
    expected_model_sha = (manifest.get("model_config") or {}).get("sha256")
    actual_model_sha = _file_sha256(model_config_path)
    if expected_model_sha and expected_model_sha != actual_model_sha:
        errors.append(
            "model config hash mismatch: "
            f"checkpoint={expected_model_sha}, current={actual_model_sha}"
        )
    expected_tokenizer_sha = (manifest.get("tokenizer") or {}).get("sha256")
    actual_tokenizer_sha = (
        _file_sha256(tokenizer_path)
        if tokenizer_path and os.path.exists(tokenizer_path)
        else None
    )
    if expected_tokenizer_sha and expected_tokenizer_sha != actual_tokenizer_sha:
        errors.append(
            "tokenizer hash mismatch: "
            f"checkpoint={expected_tokenizer_sha}, current={actual_tokenizer_sha}"
        )
    if errors:
        raise SystemExit(
            "Initialization artifact validation failed.\n- " + "\n- ".join(errors)
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint_dir": checkpoint_dir,
        "model_sha256": _file_sha256(model_path),
        "artifacts_manifest_path": os.path.abspath(manifest_path),
        "artifacts_manifest_sha256": _file_sha256(manifest_path),
        "parent_compatibility_fingerprint": manifest.get(
            "compatibility_fingerprint"
        ),
    }


def _utc_now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _git_is_dirty():
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(output.strip())
    except Exception:
        return None


def _parse_step_name(step_dir):
    if not step_dir or not step_dir.startswith("step_"):
        return None
    suffix = step_dir[len("step_") :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _find_data_meta_path(data_cfg):
    train_bin = data_cfg.get("train_bin")
    if not train_bin:
        return None
    return os.path.join(os.path.dirname(train_bin) or ".", "data_meta.json")


def _indexed_manifest_info(corpus_dir):
    if not corpus_dir:
        return None
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    manifest = _load_json_if_exists(manifest_path)
    return {
        "corpus_dir": os.path.abspath(corpus_dir),
        "manifest_path": (
            os.path.abspath(manifest_path) if os.path.exists(manifest_path) else None
        ),
        "manifest_sha256": (
            _file_sha256(manifest_path) if os.path.exists(manifest_path) else None
        ),
        "corpus_sha256": (manifest or {}).get("corpus_sha256"),
        "tokenizer_sha256": (manifest or {}).get("tokenizer_sha256"),
        "recipe_sha256": (manifest or {}).get("recipe_sha256"),
        "document_count": (manifest or {}).get("document_count"),
        "token_count": (manifest or {}).get("token_count"),
    }


def _build_artifact_manifest(model_config_path, train_config_path, model_cfg, train_cfg, data_cfg):
    tokenizer_path = _resolve_tokenizer_path(train_cfg)
    tokenizer_info = {
        "path": os.path.abspath(tokenizer_path) if tokenizer_path else None,
        "sha256": None,
        "vocab_size": None,
    }
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer_info["sha256"] = _file_sha256(tokenizer_path)
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        tokenizer_info["vocab_size"] = int(sp.vocab_size())

    data_meta_path = _find_data_meta_path(data_cfg)
    data_meta = _load_json_if_exists(data_meta_path) if data_meta_path else None
    data_meta_info = {
        "path": os.path.abspath(data_meta_path) if data_meta_path and os.path.exists(data_meta_path) else None,
        "sha256": _file_sha256(data_meta_path) if data_meta_path and os.path.exists(data_meta_path) else None,
        "tokenizer_sha256": (data_meta or {}).get("tokenizer_sha256"),
        "tokenizer_vocab_size": (data_meta or {}).get("tokenizer_vocab_size"),
    }

    model_cfg_sha = _file_sha256(model_config_path) if os.path.exists(model_config_path) else None
    train_cfg_sha = _file_sha256(train_config_path) if os.path.exists(train_config_path) else None
    indexed_cfg = data_cfg.get("indexed", {}) or {}

    return {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "git": {
            "commit": _git_head_commit(),
            "dirty": _git_is_dirty(),
        },
        "model_config": {
            "path": os.path.abspath(model_config_path),
            "sha256": model_cfg_sha,
            "vocab_size": model_cfg.get("vocab_size"),
        },
        "train_config": {
            "path": os.path.abspath(train_config_path),
            "sha256": train_cfg_sha,
        },
        "tokenizer": tokenizer_info,
        "data": {
            "train_bin": data_cfg.get("train_bin"),
            "val_bin": data_cfg.get("val_bin"),
            "hf_repo": data_cfg.get("hf_repo"),
            "data_meta": data_meta_info,
            "indexed": {
                "enabled": bool(indexed_cfg.get("enabled", False)),
                "train": _indexed_manifest_info(indexed_cfg.get("train_dir")),
                "validation": _indexed_manifest_info(indexed_cfg.get("val_dir")),
                "source_weights": indexed_cfg.get("source_weights"),
                "validation_source_weights": indexed_cfg.get(
                    "validation_source_weights"
                ),
            },
        },
    }


def _training_compatibility_fingerprint(artifact_manifest):
    """Hash only immutable inputs that must agree for an exact resume."""
    data = artifact_manifest.get("data") or {}
    indexed = data.get("indexed") or {}
    payload = {
        "model_config_sha256": (
            artifact_manifest.get("model_config") or {}
        ).get("sha256"),
        "train_config_sha256": (
            artifact_manifest.get("train_config") or {}
        ).get("sha256"),
        "tokenizer_sha256": (
            artifact_manifest.get("tokenizer") or {}
        ).get("sha256"),
        "flat_data_meta_sha256": (data.get("data_meta") or {}).get("sha256"),
        "indexed_train_corpus_sha256": (
            indexed.get("train") or {}
        ).get("corpus_sha256"),
        "indexed_validation_corpus_sha256": (
            indexed.get("validation") or {}
        ).get("corpus_sha256"),
        "source_weights": indexed.get("source_weights"),
        "validation_source_weights": indexed.get("validation_source_weights"),
        "initialization_model_sha256": (
            artifact_manifest.get("initialization") or {}
        ).get("model_sha256"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_artifact_manifest(output_dir, artifact_manifest):
    path = os.path.join(output_dir, "artifacts_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2, sort_keys=True)
    return path


def _collect_artifact_upload_files(artifact_manifest, artifact_manifest_path, model_config_path, train_config_path):
    uploads = []
    seen = set()

    def _add(local_path, remote_path):
        if not local_path or not os.path.exists(local_path):
            return
        key = (os.path.abspath(local_path), remote_path)
        if key in seen:
            return
        seen.add(key)
        uploads.append((local_path, remote_path))

    _add(artifact_manifest_path, "artifacts/artifacts_manifest.json")
    _add(model_config_path, f"artifacts/configs/{os.path.basename(model_config_path)}")
    _add(train_config_path, f"artifacts/configs/{os.path.basename(train_config_path)}")

    tokenizer_path = (artifact_manifest.get("tokenizer") or {}).get("path")
    if tokenizer_path:
        tokenizer_name = os.path.basename(tokenizer_path)
        _add(tokenizer_path, f"artifacts/tokenizer/{tokenizer_name}")
        tokenizer_dir = os.path.dirname(tokenizer_path)
        _add(os.path.join(tokenizer_dir, "spm.vocab"), "artifacts/tokenizer/spm.vocab")
        _add(os.path.join(tokenizer_dir, "tokenizer_meta.json"), "artifacts/tokenizer/tokenizer_meta.json")

    data_meta_path = (((artifact_manifest.get("data") or {}).get("data_meta") or {}).get("path"))
    if data_meta_path:
        _add(data_meta_path, "artifacts/data/data_meta.json")
    indexed_info = (artifact_manifest.get("data") or {}).get("indexed") or {}
    for split_name in ("train", "validation"):
        manifest_path = (indexed_info.get(split_name) or {}).get("manifest_path")
        if manifest_path:
            _add(
                manifest_path,
                f"artifacts/data/{split_name}_indexed_manifest.json",
            )

    return uploads


def _resolve_tokenizer_path(train_cfg):
    checks_cfg = train_cfg.get("checks", {})
    fixed_cfg = checks_cfg.get("fixed_prompt", {})
    data_prep_cfg = train_cfg.get("data_prep", {})
    eval_cfg = train_cfg.get("eval", {})
    inference_cfg = train_cfg.get("inference", {})
    candidates = [
        fixed_cfg.get("tokenizer_model"),
        data_prep_cfg.get("tokenizer_model"),
        eval_cfg.get("tokenizer_model"),
        inference_cfg.get("tokenizer"),
    ]
    for path in candidates:
        if path:
            return path
    return None


def _verify_tokenizer_compat(model_cfg, train_cfg, data_cfg):
    tokenizer_path = _resolve_tokenizer_path(train_cfg)
    if not tokenizer_path:
        return
    if not os.path.exists(tokenizer_path):
        raise SystemExit(f"Tokenizer model not found: {tokenizer_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    mismatches = []
    expected_vocab = int(model_cfg["vocab_size"])
    actual_vocab = int(sp.vocab_size())
    if actual_vocab != expected_vocab:
        mismatches.append(
            f"vocab_size mismatch: model={expected_vocab} tokenizer={actual_vocab}"
        )

    tokenizer_ids = {
        "bos_token_id": sp.bos_id(),
        "eos_token_id": sp.eos_id(),
        "pad_token_id": sp.pad_id(),
    }
    for key, tok_id in tokenizer_ids.items():
        model_id = model_cfg.get(key)
        if model_id is None or tok_id is None:
            continue
        if int(model_id) != int(tok_id):
            mismatches.append(f"{key} mismatch: model={model_id} tokenizer={tok_id}")

    if mismatches:
        details = "\n- ".join(mismatches)
        raise SystemExit(
            "Tokenizer/model config mismatch. Fix before training:\n"
            f"- {details}"
        )

    indexed_cfg = data_cfg.get("indexed", {}) or {}
    if indexed_cfg.get("enabled", False):
        current_hash = _file_sha256(tokenizer_path)
        configured_hash = indexed_cfg.get("tokenizer_sha256")
        if configured_hash and configured_hash.lower() != current_hash:
            raise SystemExit(
                "Tokenizer/indexed-data config mismatch: "
                f"configured sha256={configured_hash}, current tokenizer={current_hash}."
            )
        for split_name, corpus_dir in (
            ("train", indexed_cfg.get("train_dir")),
            ("validation", indexed_cfg.get("val_dir")),
        ):
            manifest_path = (
                os.path.join(corpus_dir, "manifest.json") if corpus_dir else None
            )
            manifest = _load_json_if_exists(manifest_path)
            if not manifest:
                raise SystemExit(
                    f"Indexed {split_name} manifest not found or invalid: {manifest_path}"
                )
            manifest_hash = manifest.get("tokenizer_sha256")
            if manifest_hash != current_hash:
                raise SystemExit(
                    f"Tokenizer/indexed {split_name} mismatch: "
                    f"manifest sha256={manifest_hash}, current tokenizer={current_hash}."
                )
        return

    train_bin = data_cfg.get("train_bin")
    if not train_bin:
        return
    meta_path = os.path.join(os.path.dirname(train_bin) or ".", "data_meta.json")
    meta = _load_json_if_exists(meta_path)
    if not meta:
        if data_cfg.get("hf_repo"):
            print(
                "[data] data_meta.json not found for train data. "
                "If you use pre-tokenized HF data, ensure tokenizer matches that dataset."
            )
        return

    meta_vocab = meta.get("tokenizer_vocab_size")
    if meta_vocab is not None and int(meta_vocab) != actual_vocab:
        raise SystemExit(
            "Tokenizer/data mismatch: data_meta tokenizer_vocab_size "
            f"is {meta_vocab}, current tokenizer vocab_size is {actual_vocab}."
        )

    meta_special = meta.get("tokenizer_special_ids") or {}
    if isinstance(meta_special, dict):
        key_pairs = (
            ("bos_token_id", "bos"),
            ("eos_token_id", "eos"),
            ("pad_token_id", "pad"),
        )
        for model_key, meta_key in key_pairs:
            if meta_key not in meta_special:
                continue
            meta_id = meta_special.get(meta_key)
            model_id = model_cfg.get(model_key)
            if meta_id is None or model_id is None:
                continue
            if int(meta_id) != int(model_id):
                raise SystemExit(
                    "Tokenizer/data mismatch: "
                    f"data_meta {meta_key}={meta_id}, model {model_key}={model_id}."
                )

    meta_hash = meta.get("tokenizer_sha256")
    if meta_hash:
        current_hash = _file_sha256(tokenizer_path)
        if current_hash != meta_hash:
            raise SystemExit(
                "Tokenizer file does not match data tokenization. "
                f"Expected sha256={meta_hash}, got {current_hash}. "
                "Re-run prepare-data with this tokenizer or switch to the matching tokenizer."
            )
    else:
        print(
            "[data] data_meta.json has no tokenizer_sha256; cannot fully verify "
            "tokenizer/data consistency. Re-run prepare-data to add tokenizer fingerprinting."
        )


class CheckpointUploader:
    """Uploads/checks/resumes checkpoints from HuggingFace Hub."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.repo_id = config.get("repo_id")
        self.upload_interval = config.get("upload_interval", 1000)
        self.upload_optimizer = config.get("upload_optimizer", False)
        self.token = config.get("token") or os.environ.get("HF_TOKEN")
        self.retry_max_attempts = max(1, int(config.get("retry_max_attempts", 3)))
        self.retry_backoff_seconds = max(1, int(config.get("retry_backoff_seconds", 5)))
        self.verify_upload = bool(config.get("verify_upload", True))
        retention_cfg = config.get("retention", {}) or {}
        self.prune_remote_enabled = bool(retention_cfg.get("enabled", True))
        self.keep_latest = max(0, int(retention_cfg.get("keep_latest", 1)))
        self.keep_best = max(0, int(retention_cfg.get("keep_best", 2)))
        self._api = None
        self._last_upload_step = 0

        if self.enabled and not self.repo_id:
            print("[checkpoint_upload] Warning: enabled but no repo_id set, disabling")
            self.enabled = False

        if self.enabled and not self.upload_optimizer:
            print(
                "[checkpoint_upload] Note: upload_optimizer=false is ignored in exact-resume mode; "
                "full Accelerate checkpoint folders are uploaded."
            )

        if self.enabled and self.token:
            try:
                hf_login(token=self.token, add_to_git_credential=False)
            except Exception as e:
                print(f"[checkpoint_upload] Warning: HF login failed: {e}")

    @property
    def api(self):
        if self._api is None:
            self._api = HfApi()
            if self.enabled:
                try:
                    self._api.create_repo(self.repo_id, repo_type="model", exist_ok=True)
                except Exception as e:
                    print(f"[checkpoint_upload] Warning: Could not create repo: {e}")
        return self._api

    def _with_retries(self, op_name, fn):
        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                print(
                    f"[checkpoint_upload] {op_name} failed on attempt "
                    f"{attempt}/{self.retry_max_attempts}: {e}"
                )
                if attempt < self.retry_max_attempts:
                    delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(delay)
        raise RuntimeError(f"{op_name} failed after retries: {last_error}") from last_error

    def should_upload(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step - self._last_upload_step >= self.upload_interval

    def _collect_relative_files(self, folder_path):
        rel_paths = []
        for root, _, files in os.walk(folder_path):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_paths.append(os.path.relpath(full_path, folder_path))
        return sorted(rel_paths)

    def _verify_remote_files(self, folder_name, relative_files):
        remote_files = set(self.api.list_repo_files(self.repo_id, repo_type="model"))
        expected = {f"{folder_name}/{rel}" for rel in relative_files}
        missing = sorted(expected - remote_files)
        return missing

    def upload(self, checkpoint_dir: str, step: int, is_final: bool = False):
        if not self.enabled:
            return False

        if not os.path.isdir(checkpoint_dir):
            print(f"[checkpoint_upload] Checkpoint directory does not exist: {checkpoint_dir}")
            return False

        folder_name = "final" if is_final else f"step_{step:07d}"
        expected_files = self._collect_relative_files(checkpoint_dir)
        if not expected_files:
            print(f"[checkpoint_upload] Nothing to upload in {checkpoint_dir}")
            return False

        try:
            self._with_retries(
                f"upload checkpoint {folder_name}",
                lambda: self.api.upload_folder(
                    folder_path=checkpoint_dir,
                    path_in_repo=folder_name,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Upload {folder_name}",
                ),
            )

            if self.verify_upload:
                missing = self._with_retries(
                    f"verify checkpoint {folder_name}",
                    lambda: self._verify_remote_files(folder_name, expected_files),
                )
                if missing:
                    raise RuntimeError(
                        f"Remote checkpoint verification failed for {folder_name}. "
                        f"Missing files: {missing[:8]}"
                    )

            print(
                f"[checkpoint_upload] Uploaded {folder_name} to {self.repo_id} "
                f"({len(expected_files)} files)"
            )
            self._last_upload_step = step
            return True
        except Exception as e:
            print(f"[checkpoint_upload] Error uploading checkpoint: {e}")
            return False

    def upload_logs(self, output_dir: str):
        if not self.enabled:
            return False

        try:
            log_files = [
                "train.log",
                "fixed_prompt_samples.txt",
                "checkpoint_manifest.json",
                "artifacts_manifest.json",
            ]
            for filename in log_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    self._with_retries(
                        f"upload log {filename}",
                        lambda filepath=filepath, filename=filename: self.api.upload_file(
                            path_or_fileobj=filepath,
                            path_in_repo=filename,
                            repo_id=self.repo_id,
                            repo_type="model",
                        ),
                    )
            print(f"[checkpoint_upload] Uploaded logs to {self.repo_id}")
            return True
        except Exception as e:
            print(f"[checkpoint_upload] Error uploading logs: {e}")
            return False

    def upload_artifacts(self, local_to_remote_files):
        if not self.enabled:
            return False
        uploaded_any = False
        try:
            for local_path, remote_path in local_to_remote_files:
                if not os.path.exists(local_path):
                    continue
                self._with_retries(
                    f"upload artifact {remote_path}",
                    lambda local_path=local_path, remote_path=remote_path: self.api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=remote_path,
                        repo_id=self.repo_id,
                        repo_type="model",
                    ),
                )
                uploaded_any = True
            if uploaded_any:
                print(f"[checkpoint_upload] Uploaded artifacts to {self.repo_id}")
            return uploaded_any
        except Exception as e:
            print(f"[checkpoint_upload] Error uploading artifacts: {e}")
            return False

    def _list_remote_step_dirs(self):
        files = self.api.list_repo_files(self.repo_id, repo_type="model")
        step_dirs = set()
        for path in files:
            top_level = path.split("/", 1)[0]
            if _parse_step_name(top_level) is not None:
                step_dirs.add(top_level)
        return sorted(step_dirs, key=lambda name: _parse_step_name(name) or -1)

    def resolve_remote_step(self, selector, manifest):
        if not self.enabled:
            return None

        selector = (selector or "").strip()
        if selector == "final":
            files = self.api.list_repo_files(self.repo_id, repo_type="model")
            if any(path.startswith("final/") for path in files):
                return "final"
            return None
        remote_steps = self._list_remote_step_dirs()
        if not remote_steps:
            return None

        if selector == "latest":
            manifest_last = manifest.get("last")
            if manifest_last in remote_steps:
                return manifest_last
            return remote_steps[-1]

        if selector == "best":
            for entry in manifest.get("best", []):
                step_dir = entry.get("step")
                if step_dir in remote_steps:
                    return step_dir
            return remote_steps[-1]

        if selector.isdigit():
            selector = f"step_{int(selector):07d}"
        if selector.startswith("step_") and selector in remote_steps:
            return selector
        return None

    def download_checkpoint(self, step_dir, output_dir):
        if not self.enabled:
            raise RuntimeError("Checkpoint uploader is disabled.")
        if step_dir == "final":
            prefix = "final/"
        else:
            prefix = f"{step_dir}/"
        files = self.api.list_repo_files(self.repo_id, repo_type="model")
        target_files = [path for path in files if path.startswith(prefix)]
        if not target_files:
            raise RuntimeError(
                f"Remote checkpoint '{step_dir}' not found in {self.repo_id}."
            )

        restore_root = os.path.join(output_dir, ".hf_resume")
        os.makedirs(restore_root, exist_ok=True)
        for remote_file in target_files:
            self._with_retries(
                f"download {remote_file}",
                lambda remote_file=remote_file: hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_file,
                    repo_type="model",
                    local_dir=restore_root,
                ),
            )

        local_path = os.path.join(restore_root, step_dir)
        if not os.path.isdir(local_path):
            raise RuntimeError(
                f"Checkpoint download completed but folder missing locally: {local_path}"
            )
        return local_path

    def prune_remote(self, manifest):
        if not self.enabled or not self.prune_remote_enabled:
            return
        if self.keep_latest <= 0 and self.keep_best <= 0:
            return

        remote_steps = self._list_remote_step_dirs()
        if not remote_steps:
            return

        keep = set()
        if self.keep_latest > 0:
            keep.update(remote_steps[-self.keep_latest :])
        if manifest.get("last"):
            keep.add(manifest["last"])
        if self.keep_best > 0:
            for entry in manifest.get("best", [])[: self.keep_best]:
                step_dir = entry.get("step")
                if step_dir in remote_steps:
                    keep.add(step_dir)
        for step_dir in (manifest.get("good_slots") or {}).values():
            if step_dir in remote_steps:
                keep.add(step_dir)

        to_delete = [step_dir for step_dir in remote_steps if step_dir not in keep]
        if not to_delete:
            return

        for step_dir in to_delete:
            try:
                self._with_retries(
                    f"delete remote {step_dir}",
                    lambda step_dir=step_dir: self.api.delete_folder(
                        path_in_repo=step_dir,
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"Prune {step_dir}",
                    ),
                )
                print(f"[checkpoint_upload] Pruned remote checkpoint {step_dir}")
            except Exception as e:
                print(f"[checkpoint_upload] Warning: failed to prune {step_dir}: {e}")


class Tee:
    """Duplicate stdout/stderr to a log file."""

    def __init__(self, log_path, stream_name="stdout"):
        self.log_path = log_path
        self.stream_name = stream_name
        self.original = getattr(sys, stream_name)
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8")
        setattr(sys, stream_name, self)

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def close(self):
        setattr(sys, self.stream_name, self.original)
        self.log_file.close()


def setup_file_logging(logging_cfg, output_dir):
    """Set up file logging if enabled. Returns cleanup function."""
    log_file = logging_cfg.get("log_file")
    if not log_file:
        return lambda: None

    if log_file == "auto":
        log_path = os.path.join(output_dir, "train.log")
    else:
        log_path = log_file

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    stdout_tee = Tee(log_path, "stdout")
    stderr_tee = Tee(log_path, "stderr")

    print(f"[logging] Writing to {log_path}")

    def cleanup():
        stdout_tee.close()
        stderr_tee.close()

    return cleanup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


TRAINING_PROGRESS_SCHEMA_VERSION = 2


def _training_progress_path(checkpoint_dir, process_index=0):
    return os.path.join(
        checkpoint_dir,
        f"training_progress_rank_{int(process_index):05d}.json",
    )


def save_training_progress(
    checkpoint_dir,
    completed_steps,
    batch_rng,
    process_index=0,
    compatibility_fingerprint=None,
    counters=None,
):
    """Persist non-Accelerate state required to select the exact next batch."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _training_progress_path(checkpoint_dir, process_index)
    payload = {
        "schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
        "completed_steps": int(completed_steps),
        "batch_rng_state": batch_rng.bit_generator.state,
        "compatibility_fingerprint": compatibility_fingerprint,
        "counters": dict(counters or {}),
    }
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)
    return path


def load_training_progress(
    checkpoint_dir,
    batch_rng,
    process_index=0,
    expected_compatibility_fingerprint=None,
    counters=None,
):
    """Restore batch sampling state and return the number of completed steps."""
    path = _training_progress_path(checkpoint_dir, process_index)
    if not os.path.exists(path):
        raise SystemExit(
            "Checkpoint lacks exact-resume training progress: "
            f"{path}. Older checkpoints may still load model weights, but cannot "
            "guarantee the same next batch or schedule."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != TRAINING_PROGRESS_SCHEMA_VERSION:
        raise SystemExit(
            "Unsupported training-progress schema in "
            f"{path}: {payload.get('schema_version')!r}"
        )
    completed_steps = payload.get("completed_steps")
    rng_state = payload.get("batch_rng_state")
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise SystemExit(f"Invalid completed_steps in training progress: {path}")
    if not isinstance(rng_state, dict):
        raise SystemExit(f"Invalid batch_rng_state in training progress: {path}")
    saved_fingerprint = payload.get("compatibility_fingerprint")
    if expected_compatibility_fingerprint is not None:
        if saved_fingerprint != expected_compatibility_fingerprint:
            raise SystemExit(
                "Checkpoint compatibility fingerprint mismatch. Refusing exact "
                "resume because model, training config, tokenizer, data, or "
                f"mixture inputs differ: {path}"
            )
    try:
        batch_rng.bit_generator.state = rng_state
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Incompatible batch RNG state in training progress: {path}: {exc}"
        ) from exc
    saved_counters = payload.get("counters") or {}
    if not isinstance(saved_counters, dict):
        raise SystemExit(f"Invalid counters in training progress: {path}")
    if counters is not None:
        counters.clear()
        counters.update(saved_counters)
    return completed_steps


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_adamw_param_groups(model, weight_decay):
    """Build AdamW groups with selective weight decay for LLMs."""
    decay_params = []
    no_decay_params = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip duplicate references (e.g., tied embeddings/lm_head).
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        lname = name.lower()
        is_no_decay = (
            param.ndim < 2
            or lname.endswith(".bias")
            or "norm" in lname
            or "embed_tokens" in lname
            or "lm_head" in lname
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if not decay_params or not no_decay_params:
        raise SystemExit(
            "AdamW param grouping failed: expected both decay and no_decay groups."
        )

    param_groups = [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    stats = {
        "decay_tensors": len(decay_params),
        "no_decay_tensors": len(no_decay_params),
        "decay_params": sum(p.numel() for p in decay_params),
        "no_decay_params": sum(p.numel() for p in no_decay_params),
    }
    return param_groups, stats


def maybe_launch_screen(enabled, session_name):
    if not enabled:
        return False
    if os.environ.get("STY") or os.environ.get("TMUX"):
        return False
    if os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or os.environ.get("WORLD_SIZE"):
        print(
            "Distributed launch detected; skipping screen auto-launch. "
            "Start the accelerate command inside screen instead."
        )
        return False
    session = session_name or f"train-{time.strftime('%Y%m%d-%H%M%S')}"
    command = [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]]

    if shutil.which("screen") is not None:
        subprocess.check_call(["screen", "-dmS", session, *command])
        print(f"Started screen session '{session}'. Attach with: screen -r {session}")
        return True

    if shutil.which("tmux") is not None:
        subprocess.check_call(["tmux", "new-session", "-d", "-s", session, *command])
        print(f"Started tmux session '{session}'. Attach with: tmux attach -t {session}")
        return True

    print("screen/tmux not found; running in foreground.")
    return False


def load_memmap(path, dtype, hf_repo=None):
    """Load a memmap file, optionally downloading from HuggingFace first."""
    if not os.path.exists(path) and hf_repo:
        # Download from HuggingFace if not present locally
        from huggingface_hub import hf_hub_download
        filename = os.path.basename(path)
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                repo_type="dataset",
                local_dir=os.path.dirname(path) or ".",
            )
            path = local_path
            print(f"[data] Downloaded missing file {filename} from {hf_repo}")
        except Exception as e:
            print(f"[data] Warning: Could not download from HF: {e}, trying local path")
    
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, batch_size, block_size, rng, device):
    # HF CausalLM loss already shifts labels internally (token < n predicts n),
    # so labels must be aligned with input_ids, not pre-shifted.
    max_idx = len(data) - block_size + 1
    if max_idx <= 0:
        raise SystemExit("Data file too small for block_size.")
    idx = rng.integers(0, max_idx, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in idx])
    x = torch.from_numpy(x).long().to(device, non_blocking=True)
    y = x.clone()
    return x, y


class StreamingDataLoader:
    """Streams tokenized data from HuggingFace with prefetching."""
    
    def __init__(
        self,
        hf_repo: str,
        split: str,
        block_size: int,
        batch_size: int,
        device,
        buffer_size: int = 10000,
        prefetch_batches: int = 10,
        seed: int = 1337,
    ):
        from datasets import load_dataset
        from threading import Thread
        from queue import Queue
        
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.buffer_size = buffer_size
        self.prefetch_batches = prefetch_batches
        self.rng = np.random.default_rng(seed)
        
        # Load streaming dataset
        print(f"[streaming] Loading {hf_repo} split={split} (streaming mode)")
        self.dataset = load_dataset(
            hf_repo,
            split=split,
            streaming=True,
        ).shuffle(seed=seed, buffer_size=buffer_size)
        
        # Token buffer - accumulates tokens from streamed examples
        self._token_buffer = np.array([], dtype=np.uint16)
        self._dataset_iter = iter(self.dataset)
        self._min_buffer_tokens = block_size * batch_size * 4  # Keep buffer well-stocked
        
        # Prefetch queue
        self._batch_queue = Queue(maxsize=prefetch_batches)
        self._stop_prefetch = False
        self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        
        print(f"[streaming] Started with buffer_size={buffer_size}, prefetch={prefetch_batches}")
    
    def _fill_buffer(self):
        """Fill token buffer from streaming dataset."""
        tokens_needed = self._min_buffer_tokens - len(self._token_buffer)
        if tokens_needed <= 0:
            return
        
        new_tokens = []
        tokens_collected = 0
        
        while tokens_collected < tokens_needed:
            try:
                example = next(self._dataset_iter)
                # Expect pre-tokenized data with 'input_ids' or 'tokens' field
                if "input_ids" in example:
                    toks = example["input_ids"]
                elif "tokens" in example:
                    toks = example["tokens"]
                else:
                    # Skip examples without tokens
                    continue
                
                if isinstance(toks, list):
                    new_tokens.extend(toks)
                    tokens_collected += len(toks)
                    
            except StopIteration:
                # Dataset exhausted, restart
                self._dataset_iter = iter(self.dataset)
        
        if new_tokens:
            self._token_buffer = np.concatenate([
                self._token_buffer,
                np.array(new_tokens, dtype=np.uint16)
            ])
    
    def _get_batch_from_buffer(self):
        """Extract a batch from the token buffer."""
        self._fill_buffer()
        
        # HF CausalLM loss shifts labels internally, so no manual +1 offset.
        batch_tokens = self.block_size
        total_needed = self.batch_size * batch_tokens
        
        if len(self._token_buffer) < total_needed:
            raise RuntimeError("Token buffer too small - this shouldn't happen")
        
        # Random starting positions within buffer
        max_start = len(self._token_buffer) - batch_tokens + 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        
        x_list = []
        for start in starts:
            x_list.append(self._token_buffer[start:start + self.block_size])
        
        x = torch.from_numpy(np.stack(x_list)).long().to(self.device, non_blocking=True)
        y = x.clone()
        
        # Trim buffer occasionally to prevent unbounded growth
        if len(self._token_buffer) > self._min_buffer_tokens * 2:
            self._token_buffer = self._token_buffer[-self._min_buffer_tokens:]
        
        return x, y
    
    def _prefetch_worker(self):
        """Background thread that prefetches batches."""
        while not self._stop_prefetch:
            try:
                batch = self._get_batch_from_buffer()
                self._batch_queue.put(batch, timeout=1.0)
            except Exception as e:
                if not self._stop_prefetch:
                    print(f"[streaming] Prefetch error: {e}")
                    import time
                    time.sleep(0.1)
    
    def get_batch(self):
        """Get next batch (from prefetch queue)."""
        return self._batch_queue.get(timeout=30.0)
    
    def stop(self):
        """Stop prefetch thread."""
        self._stop_prefetch = True
        self._prefetch_thread.join(timeout=2.0)


class IndexedDataLoader:
    """Stateless-on-RNG adapter from indexed corpora to model-ready tensors."""

    def __init__(
        self,
        corpus_dir,
        *,
        batch_size,
        block_size,
        device,
        source_weights=None,
        expected_tokenizer_sha256=None,
        expected_recipe_sha256=None,
        verify_hashes=True,
    ):
        try:
            self.reader = IndexedShardReader(
                corpus_dir,
                expected_tokenizer_sha256=expected_tokenizer_sha256,
                expected_recipe_sha256=expected_recipe_sha256,
                verify_hashes=verify_hashes,
            )
            self.sampler = IndexedCorpusSampler(
                self.reader,
                source_weights=source_weights,
            )
        except (OSError, ShardFormatError) as exc:
            raise SystemExit(f"Invalid indexed corpus {corpus_dir}: {exc}") from exc
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.device = device

    @property
    def corpus_sha256(self):
        return self.reader.manifest["corpus_sha256"]

    @property
    def source_ids(self):
        return self.sampler.source_ids

    def get_batch(self, rng, source_id=None):
        packed = self.sampler.sample_batch(
            batch_size=self.batch_size,
            block_size=self.block_size,
            rng=rng,
            source_id=source_id,
        )
        input_ids = torch.from_numpy(packed.input_ids.astype(np.int64, copy=False))
        labels = torch.from_numpy(packed.labels)
        position_ids = torch.from_numpy(packed.position_ids)
        return (
            input_ids.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
            {
                "position_ids": position_ids.to(self.device, non_blocking=True),
                # Transformers only detects reset-position packed sequences
                # when no KV cache is active.
                "use_cache": False,
            },
        )


def create_data_loader(data_cfg, batch_size, block_size, device, seed):
    """Create an indexed, streaming, or flat-memmap data loader."""
    indexed_cfg = data_cfg.get("indexed", {}) or {}
    if indexed_cfg.get("enabled", False):
        train_dir = indexed_cfg.get("train_dir")
        val_dir = indexed_cfg.get("val_dir")
        if not train_dir or not val_dir:
            raise SystemExit(
                "indexed.train_dir and indexed.val_dir are required when "
                "indexed.enabled=true"
            )
        common = {
            "batch_size": batch_size,
            "block_size": block_size,
            "device": device,
            "expected_tokenizer_sha256": indexed_cfg.get("tokenizer_sha256"),
            "expected_recipe_sha256": indexed_cfg.get("recipe_sha256"),
            "verify_hashes": indexed_cfg.get("verify_hashes", True),
        }
        train_loader = IndexedDataLoader(
            train_dir,
            source_weights=indexed_cfg.get("source_weights"),
            **common,
        )
        validation_source_weights = indexed_cfg.get("validation_source_weights")
        if validation_source_weights is None:
            validation_source_weights = indexed_cfg.get("source_weights")
        val_loader = IndexedDataLoader(
            val_dir,
            source_weights=validation_source_weights,
            **common,
        )
        print(
            "[data] Indexed corpora loaded: "
            f"train={train_loader.corpus_sha256[:12]} "
            f"val={val_loader.corpus_sha256[:12]}"
        )
        return {"mode": "indexed", "train": train_loader, "val": val_loader}

    streaming_cfg = data_cfg.get("streaming", {})
    
    if streaming_cfg.get("enabled", False):
        hf_repo = streaming_cfg.get("hf_repo")
        if not hf_repo:
            raise SystemExit("streaming.hf_repo required when streaming.enabled=true")
        
        train_loader = StreamingDataLoader(
            hf_repo=hf_repo,
            split=streaming_cfg.get("train_split", "train"),
            block_size=block_size,
            batch_size=batch_size,
            device=device,
            buffer_size=streaming_cfg.get("buffer_size", 10000),
            prefetch_batches=streaming_cfg.get("prefetch_batches", 10),
            seed=seed,
        )
        
        # For validation, still use local or download
        # (val is small, streaming overhead not worth it)
        val_data = None
        if data_cfg.get("val_bin"):
            dtype = np.dtype(data_cfg["dtype"])
            hf_data_repo = data_cfg.get("hf_repo")
            val_data = load_memmap(data_cfg["val_bin"], dtype, hf_repo=hf_data_repo)
        
        return {"mode": "streaming", "train": train_loader, "val": val_data}
    else:
        # Original memmap mode
        dtype = np.dtype(data_cfg["dtype"])
        hf_data_repo = data_cfg.get("hf_repo")
        train_data = load_memmap(data_cfg["train_bin"], dtype, hf_repo=hf_data_repo)
        val_data = load_memmap(data_cfg["val_bin"], dtype, hf_repo=hf_data_repo)
        return {"mode": "memmap", "train": train_data, "val": val_data}


def get_model_batch(data, data_mode, batch_size, block_size, rng, device):
    """Return input IDs, labels, and any model kwargs for one batch."""
    if data_mode == "indexed":
        return data.get_batch(rng)
    if data_mode == "streaming":
        x, y = data.get_batch()
        return x, y, {}
    x, y = get_batch(data, batch_size, block_size, rng, device)
    return x, y, {}


@torch.no_grad()
def evaluate(
    model,
    data,
    batch_size,
    block_size,
    rng,
    device,
    batches,
    accelerator,
    data_mode="memmap",
):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y, model_kwargs = get_model_batch(
            data,
            data_mode,
            batch_size,
            block_size,
            rng,
            device,
        )
        outputs = model(input_ids=x, labels=y, **model_kwargs)
        losses.append(outputs.loss.detach())
    loss_tensor = torch.stack(losses)
    loss_tensor = accelerator.gather(loss_tensor)
    return loss_tensor.mean().item()


@torch.no_grad()
def evaluate_indexed_by_source(
    model,
    data,
    rng,
    batches,
    accelerator,
):
    """Return held-out loss, perplexity, and evaluated tokens per source."""
    if not isinstance(data, IndexedDataLoader):
        raise TypeError("source-aware evaluation requires indexed validation data")
    model.eval()
    results = {}
    for source_id in data.source_ids:
        losses = []
        for _ in range(batches):
            x, y, model_kwargs = data.get_batch(rng, source_id=source_id)
            loss = model(input_ids=x, labels=y, **model_kwargs).loss.detach()
            losses.append(loss)
        loss_tensor = accelerator.gather(torch.stack(losses))
        mean_loss = loss_tensor.mean().item()
        results[source_id] = {
            "loss": mean_loss,
            "perplexity": math.exp(min(20, mean_loss)),
            "batches": int(batches * accelerator.num_processes),
            "tokens": int(
                batches
                * accelerator.num_processes
                * data.batch_size
                * data.block_size
            ),
        }
    return results


def maybe_apply_budget_guard(budget, tokens_per_step):
    if not budget:
        return None
    throughput_path = budget.get("throughput_path")
    if not throughput_path or not os.path.exists(throughput_path):
        return None
    with open(throughput_path, "r", encoding="utf-8") as f:
        throughput = json.load(f)
    tokens_per_sec = throughput.get("tokens_per_sec")
    if not tokens_per_sec:
        return None
    target_tokens = budget.get("target_tokens", 0)
    hourly_rate = budget.get("hourly_rate", 0.0)
    max_cost = budget.get("run_max_cost", budget.get("max_cost", 0.0))
    if target_tokens <= 0 or hourly_rate <= 0 or max_cost <= 0:
        return None
    total_hours = target_tokens / tokens_per_sec / 3600.0
    projected_cost = total_hours * hourly_rate
    if projected_cost > max_cost:
        raise SystemExit(
            f"Projected cost ${projected_cost:.2f} exceeds max_cost ${max_cost:.2f}. "
            "Reduce target_tokens or pick cheaper GPUs."
        )
    return math.ceil(target_tokens / tokens_per_step)


def _make_local_checkpoint_dir(output_dir, logical_name, mode):
    if mode == "ephemeral":
        temp_root = os.path.join(output_dir, ".tmp_checkpoints")
        os.makedirs(temp_root, exist_ok=True)
        return tempfile.mkdtemp(prefix=f"{logical_name}_", dir=temp_root)
    return os.path.join(output_dir, logical_name)


def _cleanup_dir(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _persist_ephemeral_checkpoint_dir(checkpoint_dir, output_dir, logical_name):
    fallback_dir = os.path.join(output_dir, f"{logical_name}_upload_failed")
    if os.path.exists(fallback_dir):
        shutil.rmtree(fallback_dir)
    shutil.move(checkpoint_dir, fallback_dir)
    return fallback_dir


def rotate_checkpoints(output_dir, limit, protected=None):
    """Keep at most `limit` total checkpoints, deleting oldest unprotected first."""
    if limit <= 0:
        return
    protected = set(protected or [])
    entries = [d for d in os.listdir(output_dir) if d.startswith("step_")]
    total_count = len(entries)
    if total_count <= limit:
        return
    # Delete oldest unprotected checkpoints until we have at most `limit` total
    unprotected = sorted([e for e in entries if e not in protected])
    to_remove_count = total_count - limit
    to_remove = unprotected[:to_remove_count]
    for name in to_remove:
        path = os.path.join(output_dir, name)
        shutil.rmtree(path)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _load_yaml_or_json(path):
    if path.endswith(".json"):
        return _load_json(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_nested(cfg, dotted_path, value):
    parts = dotted_path.split(".")
    current = cfg
    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[parts[-1]] = value


def _get_nested(cfg, dotted_path, default=None):
    parts = dotted_path.split(".")
    current = cfg
    for key in parts:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_prompt_list(path):
    if not path or not os.path.exists(path):
        return []
    data = _load_yaml_or_json(path)
    if isinstance(data, dict):
        data = data.get("prompts", [])
    if not isinstance(data, list):
        return []
    return [str(prompt) for prompt in data if prompt is not None]


def _collect_prompts(fixed_cfg):
    prompts = []
    prompt_list_path = fixed_cfg.get("prompt_list_path")
    prompts.extend(_load_prompt_list(prompt_list_path))
    prompts.extend(fixed_cfg.get("prompt_list", []) or [])
    if not prompts:
        prompts = [fixed_cfg.get("prompt", "The quick brown fox")]
    return [str(prompt) for prompt in prompts]


def _format_sample_block(step, prompt, sample, tag=None):
    header = f"step {step}"
    if tag:
        header = f"{header} [{tag}]"
    return f"{header}\nprompt: {prompt}\n{sample}\n"


def _read_command_queue(path, start_offset):
    if not path or not os.path.exists(path):
        return [], start_offset
    commands = []
    with open(path, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                commands.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        new_offset = f.tell()
    return commands, new_offset


class RuntimeControl:
    def __init__(self, cfg, output_dir):
        self.enabled = bool(cfg.get("enabled", False))
        self.poll_interval_steps = max(1, int(cfg.get("poll_interval_steps", 50)))
        self.control_path = cfg.get("control_path") or os.path.join(output_dir, "runtime_control.yaml")
        self.command_path = cfg.get("command_path") or os.path.join(output_dir, "commands.jsonl")
        self.cursor_path = cfg.get("cursor_path") or os.path.join(
            output_dir, "runtime_cursor.json"
        )
        self.allowed_updates = set(cfg.get("allowed_updates", []))
        self._last_control_mtime = None
        self._command_offset = 0
        if self.enabled and os.path.exists(self.cursor_path):
            try:
                cursor = _load_json(self.cursor_path)
                self._command_offset = max(
                    0, int(cursor.get("command_offset", 0))
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self._command_offset = 0

    def poll(self, step):
        if not self.enabled or step % self.poll_interval_steps != 0:
            return {}, []
        updates = {}
        if self.control_path and os.path.exists(self.control_path):
            mtime = os.path.getmtime(self.control_path)
            if self._last_control_mtime is None or mtime > self._last_control_mtime:
                payload = _load_yaml_or_json(self.control_path) or {}
                if isinstance(payload, dict):
                    updates.update(payload.get("updates", {}))
                    if "prompts" in payload:
                        updates["checks.fixed_prompt.prompt_list"] = payload["prompts"]
                    if "prompt_list_path" in payload:
                        updates["checks.fixed_prompt.prompt_list_path"] = payload["prompt_list_path"]
                self._last_control_mtime = mtime
        commands, self._command_offset = _read_command_queue(self.command_path, self._command_offset)
        atomic_write_json(
            Path(self.cursor_path),
            {
                "schema_version": 1,
                "command_path": os.path.abspath(self.command_path),
                "command_offset": self._command_offset,
                "updated_at": _utc_now_iso(),
            },
        )
        return updates, commands


class MetricsLogger:
    def __init__(self, cfg, output_dir, is_main_process):
        self.enabled = bool(cfg.get("enabled", True)) and is_main_process
        self.console_summary = bool(cfg.get("console_summary", True))
        self.tb = None
        self.wandb = None
        if not self.enabled:
            return
        tb_cfg = cfg.get("tensorboard", {})
        if tb_cfg.get("enabled", False):
            log_dir = tb_cfg.get("log_dir") or os.path.join(output_dir, "tb")
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise SystemExit("TensorBoard enabled but not installed.") from exc
            self.tb = SummaryWriter(log_dir=log_dir)
        wandb_cfg = cfg.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            try:
                import wandb
            except ImportError as exc:
                raise SystemExit("wandb enabled but not installed.") from exc
            self.wandb = wandb.init(
                project=wandb_cfg.get("project"),
                name=wandb_cfg.get("name"),
                entity=wandb_cfg.get("entity"),
                tags=wandb_cfg.get("tags"),
                group=wandb_cfg.get("group"),
                config=wandb_cfg.get("config"),
            )

    def log_metrics(self, step, metrics):
        if not self.enabled:
            return
        if self.tb:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb.add_scalar(key, value, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_text(self, step, key, text):
        if not self.enabled:
            return
        if self.tb:
            self.tb.add_text(key, text, step)
        if self.wandb:
            self.wandb.log({key: text}, step=step)

    def maybe_print(self, text):
        if self.enabled and self.console_summary:
            print(text)


def _get_gpu_stats():
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "gpu_mem_free_gb": free_bytes / (1024**3),
        "gpu_mem_used_gb": used_bytes / (1024**3),
        "gpu_mem_total_gb": total_bytes / (1024**3),
        "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def _load_checkpoint_manifest(output_dir):
    manifest_path = os.path.join(output_dir, "checkpoint_manifest.json")
    if not os.path.exists(manifest_path):
        return {"last": None, "best": [], "good_slots": {}, "steps": {}, "final": None}
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("last", None)
    manifest.setdefault("best", [])
    manifest.setdefault("good_slots", {})
    manifest.setdefault("steps", {})
    manifest.setdefault("final", None)
    return manifest


def _save_checkpoint_manifest(output_dir, manifest):
    manifest_path = os.path.join(output_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _update_best_slots(manifest, step_dir, val_loss, max_slots):
    if val_loss is None:
        return
    best = manifest.get("best", [])
    best.append({"step": step_dir, "val_loss": val_loss})
    best = sorted(best, key=lambda item: item["val_loss"])[: max_slots]
    manifest["best"] = best


def _protected_steps(manifest):
    protected = set()
    if manifest.get("last"):
        protected.add(manifest["last"])
    for entry in manifest.get("best", []):
        if entry.get("step"):
            protected.add(entry["step"])
    for step_dir in (manifest.get("good_slots") or {}).values():
        if step_dir:
            protected.add(step_dir)
    return protected


def _resolve_slot_step(manifest, slot):
    if slot == "last":
        return manifest.get("last")
    if slot == "final":
        return "final"
    if slot == "best":
        best = manifest.get("best", [])
        if best:
            return best[0].get("step")
    good_slots = manifest.get("good_slots") or {}
    if slot in good_slots:
        return good_slots.get(slot)
    return None


def _apply_runtime_updates(optim_cfg, checks_cfg, updates):
    applied = {}
    for path, value in updates.items():
        if path.startswith("training."):
            key = path.replace("training.", "", 1)
            _set_nested(optim_cfg, key, value)
            applied[path] = value
        elif path.startswith("checks."):
            key = path.replace("checks.", "", 1)
            _set_nested(checks_cfg, key, value)
            applied[path] = value
    return applied


def _filter_allowed_updates(updates, allowed):
    if not allowed:
        return updates
    return {path: value for path, value in updates.items() if path in allowed}


def _normalize_runtime_values(optim_cfg):
    for key in ("eval_interval", "save_interval", "log_interval"):
        if key in optim_cfg:
            optim_cfg[key] = max(1, int(optim_cfg[key]))
    if "learning_rate" in optim_cfg:
        optim_cfg["learning_rate"] = float(optim_cfg["learning_rate"])


def _parse_runtime_commands(commands, good_slots):
    updates = {}
    actions = {
        "sample_prompts": [],
        "pin_slots": [],
        "stop_training": False,
        "save_now": False,
    }
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        cmd_type = cmd.get("cmd")
        if cmd_type == "set":
            path = cmd.get("path")
            if path:
                updates[path] = cmd.get("value")
        elif cmd_type == "sample_prompt":
            actions["sample_prompts"].append(cmd)
        elif cmd_type == "pin_checkpoint":
            slot = cmd.get("slot", "good_1")
            if slot in good_slots:
                actions["pin_slots"].append(cmd)
        elif cmd_type == "stop_training":
            actions["stop_training"] = True
            actions["save_now"] = bool(cmd.get("save", True))
    return updates, actions

def _truncate_at_eos(token_ids, eos_id):
    if eos_id is None:
        return token_ids
    try:
        eos_idx = token_ids.index(eos_id)
    except ValueError:
        return token_ids
    return token_ids[:eos_idx]


def _decode_tokens(sp, token_ids, special_ids, eos_id):
    token_ids = _truncate_at_eos(token_ids, eos_id)
    if special_ids:
        token_ids = [t for t in token_ids if t not in special_ids]
    return sp.decode(token_ids)


def _capture_rng_state():
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state


def _restore_rng_state(state):
    if state is None:
        return
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def sample_generate(
    model,
    sp,
    prompt,
    device,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    min_new_tokens,
    special_ids,
    eos_id,
    deterministic,
    seed,
):
    model.eval()
    input_ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_ids], device=device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
    }
    if not deterministic:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k
    if repetition_penalty and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if min_new_tokens and min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = min_new_tokens

    rng_state = None
    if seed is not None and not deterministic:
        rng_state = _capture_rng_state()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    outputs = model.generate(input_ids=input_ids, **gen_kwargs)

    if rng_state is not None:
        _restore_rng_state(rng_state)

    return _decode_tokens(sp, outputs[0].tolist(), special_ids, eos_id)


def run_overfit_microset(
    model,
    train_data,
    data_mode,
    data_cfg,
    optim_cfg,
    accelerator,
    check_cfg,
):
    if not check_cfg or not check_cfg.get("enabled", False):
        return

    block_size = data_cfg["block_size"]
    micro_batch = check_cfg.get("micro_batch_size", optim_cfg["micro_batch_size"])
    grad_accum = check_cfg.get("grad_accum_steps", optim_cfg["grad_accum_steps"])
    steps = check_cfg.get("steps", 100)
    eval_batches = check_cfg.get("eval_batches", 20)
    log_interval = max(1, check_cfg.get("log_interval", 10))
    learning_rate = check_cfg.get("learning_rate", optim_cfg["learning_rate"])
    max_grad_norm = check_cfg.get("max_grad_norm", optim_cfg["max_grad_norm"])
    warmup_steps = check_cfg.get("warmup_steps", 0)
    min_drop = check_cfg.get("min_drop", 0.3)
    target_loss = check_cfg.get("target_loss")

    eval_seed = check_cfg.get("eval_seed", optim_cfg["seed"] + 12345)
    fixed_batch = None
    if data_mode == "memmap":
        microset_tokens = int(check_cfg.get("tokens", 5_000_000))
        microset_tokens = min(microset_tokens, len(train_data))
        if microset_tokens <= block_size + 1:
            raise SystemExit("Overfit microset too small for block_size.")
        microset_data = train_data[:microset_tokens]
        if accelerator.is_main_process:
            print(
                f"[overfit check] tokens={microset_tokens} steps={steps} "
                f"micro_batch={micro_batch} grad_accum={grad_accum}"
            )
        initial_loss = evaluate(
            model,
            microset_data,
            micro_batch,
            block_size,
            np.random.default_rng(eval_seed),
            accelerator.device,
            eval_batches,
            accelerator,
        )
    else:
        fixed_batch = get_model_batch(
            train_data,
            data_mode,
            micro_batch,
            block_size,
            np.random.default_rng(eval_seed),
            accelerator.device,
        )
        if accelerator.is_main_process:
            print(
                f"[overfit check] fixed {data_mode} batch steps={steps} "
                f"micro_batch={micro_batch} grad_accum={grad_accum}"
            )
        model.eval()
        with torch.no_grad():
            x, y, model_kwargs = fixed_batch
            initial_loss_tensor = model(
                input_ids=x,
                labels=y,
                **model_kwargs,
            ).loss.detach().reshape(1)
            initial_loss = accelerator.gather(initial_loss_tensor).mean().item()

    overfit_param_groups, _ = build_adamw_param_groups(model, optim_cfg["weight_decay"])
    optimizer = torch.optim.AdamW(
        overfit_param_groups,
        lr=learning_rate,
        betas=tuple(optim_cfg["betas"]),
    )
    lr_scheduler = None
    if warmup_steps and warmup_steps > 0:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps,
        )

    rng = np.random.default_rng(optim_cfg["seed"] + 4242 + accelerator.process_index)
    for step in range(1, steps + 1):
        model.train()
        step_loss = 0.0
        for _ in range(grad_accum):
            if fixed_batch is None:
                x, y = get_batch(
                    microset_data,
                    micro_batch,
                    block_size,
                    rng,
                    accelerator.device,
                )
                model_kwargs = {}
            else:
                x, y, model_kwargs = fixed_batch
            with accelerator.accumulate(model):
                outputs = model(input_ids=x, labels=y, **model_kwargs)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()
            step_loss += loss.item()

        if accelerator.is_main_process and step % log_interval == 0:
            avg_loss = step_loss / grad_accum
            print(f"[overfit check] step {step}/{steps} loss={avg_loss:.4f}")

    if fixed_batch is None:
        final_loss = evaluate(
            model,
            microset_data,
            micro_batch,
            block_size,
            np.random.default_rng(eval_seed),
            accelerator.device,
            eval_batches,
            accelerator,
        )
    else:
        model.eval()
        with torch.no_grad():
            x, y, model_kwargs = fixed_batch
            final_loss_tensor = model(
                input_ids=x,
                labels=y,
                **model_kwargs,
            ).loss.detach().reshape(1)
            final_loss = accelerator.gather(final_loss_tensor).mean().item()
    if accelerator.is_main_process:
        drop = (initial_loss - final_loss) / max(1e-6, initial_loss)
        print(
            f"[overfit check] initial_loss={initial_loss:.4f} "
            f"final_loss={final_loss:.4f} drop={drop:.2%}"
        )
        drop_ok = min_drop is None or min_drop <= 0 or drop >= min_drop
        loss_ok = target_loss is None or final_loss <= target_loss
        if not (drop_ok or loss_ok):
            raise SystemExit(
                "Overfit microset check failed. Loss did not drop enough; "
                "verify data pipeline, labels, or masking."
            )

    accelerator.wait_for_everyone()


def setup_fixed_prompt_sampler(fixed_cfg, model_cfg, output_dir, smoke):
    if not fixed_cfg or not fixed_cfg.get("enabled", False):
        return None
    if smoke and not fixed_cfg.get("run_on_smoke", False):
        return None
    tokenizer_model = fixed_cfg.get("tokenizer_model")
    if not tokenizer_model:
        raise SystemExit("checks.fixed_prompt.tokenizer_model is required.")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)
    special_ids = {
        model_cfg["bos_token_id"],
        model_cfg["eos_token_id"],
        model_cfg["pad_token_id"],
        sp.unk_id(),
    }
    special_ids = {token_id for token_id in special_ids if token_id is not None}
    output_path = fixed_cfg.get("output_path")
    if not output_path:
        output_path = os.path.join(output_dir, "fixed_prompt_samples.txt")
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return {
        "sp": sp,
        "special_ids": special_ids,
        "output_path": output_path,
    }


def run_fixed_prompt_sample(
    model,
    accelerator,
    fixed_cfg,
    sampler_state,
    step,
    device,
    logger=None,
    prompts_override=None,
    tag="fixed_prompt",
):
    if sampler_state is None or not fixed_cfg.get("enabled", False):
        return
    if not accelerator.is_main_process:
        return
    max_new_tokens = fixed_cfg.get("max_new_tokens", 120)
    min_new_tokens = fixed_cfg.get("min_new_tokens", 16)
    temperature = fixed_cfg.get("temperature", 0.7)
    top_p = fixed_cfg.get("top_p", 0.9)
    top_k = fixed_cfg.get("top_k", 50)
    repetition_penalty = fixed_cfg.get("repetition_penalty", 1.1)
    deterministic = fixed_cfg.get("deterministic", True)
    seed = fixed_cfg.get("seed")

    prompts = prompts_override or _collect_prompts(fixed_cfg)
    if not prompts:
        return

    unwrapped = accelerator.unwrap_model(model)
    output_blocks = []
    for idx, prompt in enumerate(prompts, start=1):
        sample = sample_generate(
            unwrapped,
            sampler_state["sp"],
            prompt,
            device,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_new_tokens,
            sampler_state["special_ids"],
            unwrapped.config.eos_token_id,
            deterministic,
            seed,
        )
        print(f"[{tag}] step {step} prompt={prompt!r}")
        print(sample)
        output_blocks.append(_format_sample_block(step, prompt, sample, tag=tag))
        if logger is not None:
            logger.log_text(step, f"samples/{tag}/{idx}", _format_sample_block(step, prompt, sample))

    output_path = sampler_state["output_path"]
    with open(output_path, "a", encoding="utf-8") as f:
        for block in output_blocks:
            f.write(block)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Train a configured Llama-style language model.")
    parser.add_argument("--model_config", default="configs/model_100m.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument(
        "--initialize_from",
        default=None,
        help=(
            "Load model.pt from a fingerprint-validated base checkpoint while "
            "starting a new optimizer and scheduler."
        ),
    )
    parser.add_argument("--resume_from_slot", default=None)
    parser.add_argument(
        "--resume_from_hf",
        default=None,
        help="Resume from remote HF checkpoint selector: latest|best|final|step_XXXXXXX",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--launch_screen", action="store_true")
    parser.add_argument("--screen_name", default=None)
    args = parser.parse_args()

    if maybe_launch_screen(args.launch_screen, args.screen_name):
        return

    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.train_config)
    data_cfg = train_cfg["data"]
    optim_cfg = train_cfg["training"]
    budget_cfg = train_cfg.get("budget", {})
    checks_cfg = train_cfg.get("checks", {})
    logging_cfg = train_cfg.get("logging", {})
    runtime_cfg = train_cfg.get("runtime_control", {})
    observability_cfg = train_cfg.get("observability", {})
    checkpoint_cfg = train_cfg.get("checkpoint_slots", {})
    checkpoint_upload_cfg = train_cfg.get("checkpoint_upload", {})
    overfit_cfg = checks_cfg.get("overfit_microset", {})
    fixed_prompt_cfg = checks_cfg.setdefault("fixed_prompt", {})
    source_eval_cfg = checks_cfg.get("source_eval", {}) or {}

    _verify_tokenizer_compat(model_cfg, train_cfg, data_cfg)
    _normalize_runtime_values(optim_cfg)

    set_seed(optim_cfg["seed"])
    if optim_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    grad_accum = int(optim_cfg["grad_accum_steps"])
    accelerator = Accelerator(
        mixed_precision=optim_cfg.get("precision", "bf16"),
        gradient_accumulation_steps=grad_accum,
    )
    device = accelerator.device

    # Create data loader (memmap or streaming)
    data_loader = create_data_loader(
        data_cfg,
        batch_size=optim_cfg["micro_batch_size"],
        block_size=data_cfg["block_size"],
        device=device,
        seed=optim_cfg["seed"],
    )
    data_mode = data_loader["mode"]
    streaming_mode = data_mode == "streaming"
    train_data = data_loader["train"]
    val_data = data_loader["val"]

    config = LlamaConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg["num_key_value_heads"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rms_norm_eps=model_cfg["rms_norm_eps"],
        rope_theta=model_cfg["rope_theta"],
        hidden_act=model_cfg["hidden_act"],
        attention_bias=model_cfg["attention_bias"],
        mlp_bias=model_cfg["mlp_bias"],
        tie_word_embeddings=model_cfg["tie_word_embeddings"],
        pad_token_id=model_cfg["pad_token_id"],
        bos_token_id=model_cfg["bos_token_id"],
        eos_token_id=model_cfg["eos_token_id"],
    )
    model = LlamaForCausalLM(config)
    initialize_from = args.initialize_from or optim_cfg.get("initialize_from")
    initialization_lineage = None
    if initialize_from:
        initialization_lineage = _load_initial_weights(
            model,
            initialize_from,
            model_config_path=args.model_config,
            tokenizer_path=_resolve_tokenizer_path(train_cfg),
        )
        if accelerator.is_main_process:
            print(
                "[initialize] Loaded base weights "
                f"{initialization_lineage['model_sha256'][:12]} "
                f"from {initialization_lineage['checkpoint_dir']}"
            )
    if optim_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    param_count = count_parameters(model)
    if accelerator.is_main_process:
        print(f"Model parameters: {param_count/1e6:.2f}M")

    block_size = data_cfg["block_size"]
    micro_batch = optim_cfg["micro_batch_size"]
    world_size = accelerator.num_processes
    tokens_per_step = micro_batch * block_size * grad_accum * world_size
    target_steps = maybe_apply_budget_guard(budget_cfg, tokens_per_step)

    max_steps = optim_cfg["max_steps"]
    if target_steps:
        max_steps = min(max_steps, target_steps)
    if args.smoke:
        max_steps = min(max_steps, 50)

    output_dir = optim_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    artifact_manifest = _build_artifact_manifest(
        args.model_config,
        args.train_config,
        model_cfg,
        train_cfg,
        data_cfg,
    )
    if initialization_lineage is not None:
        artifact_manifest["initialization"] = initialization_lineage
    compatibility_fingerprint = _training_compatibility_fingerprint(
        artifact_manifest
    )
    artifact_manifest["compatibility_fingerprint"] = compatibility_fingerprint
    artifact_manifest_path = None
    if accelerator.is_main_process:
        artifact_manifest_path = _save_artifact_manifest(output_dir, artifact_manifest)

    # Set up file logging (only on main process)
    cleanup_logging = lambda: None
    if accelerator.is_main_process:
        cleanup_logging = setup_file_logging(logging_cfg, output_dir)

    if not runtime_cfg.get("allowed_updates"):
        runtime_cfg["allowed_updates"] = [
            "training.eval_interval",
            "training.save_interval",
            "training.log_interval",
            "training.learning_rate",
            "checks.fixed_prompt.enabled",
            "checks.fixed_prompt.deterministic",
            "checks.fixed_prompt.max_new_tokens",
            "checks.fixed_prompt.min_new_tokens",
            "checks.fixed_prompt.temperature",
            "checks.fixed_prompt.top_p",
            "checks.fixed_prompt.top_k",
            "checks.fixed_prompt.repetition_penalty",
            "checks.fixed_prompt.prompt",
            "checks.fixed_prompt.prompt_list",
            "checks.fixed_prompt.prompt_list_path",
        ]

    runtime_control = RuntimeControl(runtime_cfg, output_dir)
    metrics_logger = MetricsLogger(logging_cfg, output_dir, accelerator.is_main_process)
    run_observer = RunObserver(
        observability_cfg,
        output_dir,
        is_main_process=accelerator.is_main_process,
        budget=budget_cfg,
    )
    manifest = _load_checkpoint_manifest(output_dir)
    best_slots = max(0, int(checkpoint_cfg.get("best", 2)))
    good_slots = list(checkpoint_cfg.get("good", ["good_1", "good_2"]) or [])
    checkpoint_uploader = CheckpointUploader(checkpoint_upload_cfg) if accelerator.is_main_process else None
    local_checkpoint_mode = str(checkpoint_upload_cfg.get("local_checkpoint_mode", "persistent")).lower()
    if local_checkpoint_mode not in {"persistent", "ephemeral"}:
        raise SystemExit(
            "checkpoint_upload.local_checkpoint_mode must be one of: persistent, ephemeral"
        )
    keep_local_final = bool(
        checkpoint_upload_cfg.get("keep_local_final", local_checkpoint_mode != "ephemeral")
    )
    if local_checkpoint_mode == "ephemeral" and not (
        checkpoint_uploader and checkpoint_uploader.enabled
    ):
        raise SystemExit(
            "local_checkpoint_mode=ephemeral requires checkpoint_upload.enabled=true "
            "with a valid repo_id."
        )
    artifact_upload_files = []
    if accelerator.is_main_process:
        artifact_upload_files = _collect_artifact_upload_files(
            artifact_manifest,
            artifact_manifest_path,
            args.model_config,
            args.train_config,
        )
    if checkpoint_uploader and checkpoint_uploader.enabled and artifact_upload_files:
        artifacts_ok = checkpoint_uploader.upload_artifacts(artifact_upload_files)
        if not artifacts_ok:
            raise SystemExit(
                "Failed to upload required run artifacts to HuggingFace. "
                "Aborting to avoid unreproducible checkpoints."
            )

    model_prepared = False
    if overfit_cfg.get("enabled", False) and (not args.smoke or overfit_cfg.get("run_on_smoke", False)):
        model = accelerator.prepare(model)
        model_prepared = True
        run_overfit_microset(
            model,
            train_data,
            data_mode,
            data_cfg,
            optim_cfg,
            accelerator,
            overfit_cfg,
        )
    elif accelerator.is_main_process and overfit_cfg.get("enabled", False) and args.smoke:
        print("[overfit check] skipped on smoke run.")

    adamw_param_groups, adamw_stats = build_adamw_param_groups(model, optim_cfg["weight_decay"])
    if accelerator.is_main_process:
        print(
            "[optimizer] AdamW groups: "
            f"decay={adamw_stats['decay_tensors']} tensors "
            f"({adamw_stats['decay_params'] / 1e6:.2f}M params), "
            f"no_decay={adamw_stats['no_decay_tensors']} tensors "
            f"({adamw_stats['no_decay_params'] / 1e6:.2f}M params)"
        )
    optimizer = torch.optim.AdamW(
        adamw_param_groups,
        lr=optim_cfg["learning_rate"],
        betas=tuple(optim_cfg["betas"]),
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optim_cfg["warmup_steps"],
        num_training_steps=max_steps,
    )

    if model_prepared:
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        model_prepared = True

    fixed_prompt_sampler = setup_fixed_prompt_sampler(
        fixed_prompt_cfg,
        model_cfg,
        output_dir,
        args.smoke,
    )

    rng = np.random.default_rng(optim_cfg["seed"] + accelerator.process_index)
    completed_steps = 0
    resume_path = args.resume_from
    resume_from_hf = args.resume_from_hf or checkpoint_upload_cfg.get("resume_from_hf")
    if args.resume_from_slot:
        slot_step = _resolve_slot_step(manifest, args.resume_from_slot)
        if not slot_step:
            raise SystemExit(f"resume_from_slot '{args.resume_from_slot}' not found in manifest.")
        resume_path = os.path.join(output_dir, slot_step)
    if resume_path and not os.path.isdir(resume_path):
        resume_basename = os.path.basename(os.path.normpath(resume_path))
        can_try_hf = resume_basename == "final" or _parse_step_name(resume_basename) is not None
        if checkpoint_uploader and checkpoint_uploader.enabled and can_try_hf:
            resume_path = checkpoint_uploader.download_checkpoint(resume_basename, output_dir)
            print(f"[resume] Downloaded remote checkpoint '{resume_basename}' to {resume_path}")
        else:
            raise SystemExit(f"Resume checkpoint not found locally: {resume_path}")
    if not resume_path and resume_from_hf:
        if not (checkpoint_uploader and checkpoint_uploader.enabled):
            raise SystemExit(
                "resume_from_hf requested but checkpoint_upload is not enabled/configured."
            )
        resolved_step = checkpoint_uploader.resolve_remote_step(resume_from_hf, manifest)
        if not resolved_step:
            raise SystemExit(
                f"Could not resolve remote checkpoint selector '{resume_from_hf}' "
                f"in repo '{checkpoint_uploader.repo_id}'."
            )
        resume_path = checkpoint_uploader.download_checkpoint(resolved_step, output_dir)
        print(f"[resume] Downloaded remote checkpoint '{resolved_step}' to {resume_path}")
    if resume_path:
        print(f"[resume] Loading state from {resume_path}")
        accelerator.load_state(resume_path)
        if streaming_mode:
            raise SystemExit(
                "Exact resume is not yet supported for the streaming loader because "
                "its iterator and token buffer are not checkpointed."
            )
        restored_counters = {}
        completed_steps = load_training_progress(
            resume_path,
            rng,
            process_index=accelerator.process_index,
            expected_compatibility_fingerprint=compatibility_fingerprint,
            counters=restored_counters,
        )
        print(
            f"[resume] Restored batch RNG after {completed_steps} completed steps; "
            f"next step is {completed_steps + 1}."
        )

    tokens_processed = completed_steps * tokens_per_step
    supervised_tokens_processed = int(
        (restored_counters if resume_path else {}).get("supervised_tokens", 0)
    )
    session_tokens_processed = 0
    session_supervised_tokens = 0
    start_time = time.time()
    last_val_loss = None
    last_grad_norm = None
    stop_training = False
    stop_reason = None
    run_observer.start(
        max_steps=max_steps,
        completed_steps=completed_steps,
        tokens_processed=tokens_processed,
        compatibility_fingerprint=compatibility_fingerprint,
        experiment=train_cfg.get("experiment"),
    )

    for step in range(completed_steps + 1, max_steps + 1):
        updates, commands = runtime_control.poll(step)
        cmd_updates, actions = _parse_runtime_commands(commands, good_slots)
        updates.update(cmd_updates)
        updates = _filter_allowed_updates(updates, runtime_control.allowed_updates)
        applied = _apply_runtime_updates(optim_cfg, checks_cfg, updates)
        if applied:
            _normalize_runtime_values(optim_cfg)
            if "training.learning_rate" in applied:
                new_lr = float(optim_cfg["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr
                if hasattr(lr_scheduler, "base_lrs"):
                    lr_scheduler.base_lrs = [new_lr for _ in lr_scheduler.base_lrs]
            if accelerator.is_main_process:
                metrics_logger.maybe_print(f"[runtime] applied updates: {applied}")
                run_observer.event("runtime_updates_applied", updates=applied, step=step)

        save_requested = False
        if actions["sample_prompts"]:
            for cmd in actions["sample_prompts"]:
                prompts = cmd.get("prompts")
                if prompts is None and cmd.get("prompt") is not None:
                    prompts = [cmd.get("prompt")]
                if not prompts:
                    continue
                temp_cfg = dict(fixed_prompt_cfg)
                temp_cfg.update(cmd.get("params", {}) or {})
                run_fixed_prompt_sample(
                    model,
                    accelerator,
                    temp_cfg,
                    fixed_prompt_sampler,
                    step,
                    device,
                    logger=metrics_logger,
                    prompts_override=prompts,
                    tag=cmd.get("tag", "ad_hoc"),
                )
        if actions["pin_slots"] and accelerator.is_main_process:
            for cmd in actions["pin_slots"]:
                slot = cmd.get("slot", "good_1")
                target = cmd.get("step", "last")
                if isinstance(target, int):
                    step_dir = f"step_{target:07d}"
                elif isinstance(target, str) and target.isdigit():
                    step_dir = f"step_{int(target):07d}"
                elif target == "last":
                    step_dir = manifest.get("last")
                else:
                    step_dir = target
                if not step_dir:
                    continue
                local_exists = os.path.isdir(os.path.join(output_dir, step_dir))
                known_step = step_dir in (manifest.get("steps") or {}) or step_dir == "final"
                if not local_exists and not known_step:
                    continue
                manifest.setdefault("good_slots", {})[slot] = step_dir
                _save_checkpoint_manifest(output_dir, manifest)
                metrics_logger.maybe_print(f"[checkpoint] pinned {slot} -> {step_dir}")
        if actions["stop_training"]:
            stop_training = True
            save_requested = actions["save_now"]
            stop_reason = "runtime stop requested"
            if accelerator.is_main_process:
                run_observer.event(
                    "runtime_stop_requested",
                    step=step,
                    save_now=save_requested,
                )

        model.train()
        step_loss = 0.0
        step_supervised_tokens_local = 0
        for _ in range(grad_accum):
            x, y, model_kwargs = get_model_batch(
                train_data,
                data_mode,
                micro_batch,
                block_size,
                rng,
                device,
            )
            step_supervised_tokens_local += int((y != -100).sum().item())
            with accelerator.accumulate(model):
                outputs = model(input_ids=x, labels=y, **model_kwargs)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = None
                    if optim_cfg["max_grad_norm"] > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), optim_cfg["max_grad_norm"]
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if grad_norm is not None:
                        last_grad_norm = float(grad_norm)
            step_loss += loss.item()

        tokens_processed += tokens_per_step
        session_tokens_processed += tokens_per_step
        supervised_count = torch.tensor(
            step_supervised_tokens_local,
            device=device,
            dtype=torch.int64,
        )
        supervised_count = accelerator.reduce(supervised_count, reduction="sum")
        step_supervised_tokens = int(supervised_count.item())
        supervised_tokens_processed += step_supervised_tokens
        session_supervised_tokens += step_supervised_tokens
        completed_steps = step
        if accelerator.is_main_process and step % optim_cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            tps = session_tokens_processed / max(1e-6, elapsed)
            supervised_tps = session_supervised_tokens / max(1e-6, elapsed)
            avg_loss = step_loss / grad_accum
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            metrics = {
                "train/loss": avg_loss,
                "train/lr": current_lr,
                "train/tokens_per_sec": tps,
                "train/tokens": tokens_processed,
                "train/supervised_tokens": supervised_tokens_processed,
                "train/supervised_tokens_per_sec": supervised_tps,
            }
            if last_grad_norm is not None:
                metrics["train/grad_norm"] = last_grad_norm
            metrics.update(_get_gpu_stats())
            metrics_logger.log_metrics(step, metrics)
            run_observer.metrics(step, metrics)
            metrics_logger.maybe_print(
                f"step {step}/{max_steps} loss={avg_loss:.4f} "
                f"lr={current_lr:.2e} tps={tps:.0f} "
                f"target_tps={supervised_tps:.0f}"
            )

        if step % optim_cfg["eval_interval"] == 0:
            eval_rng = np.random.default_rng(
                int(optim_cfg["seed"])
                + accelerator.process_index
                + step * 1_000_003
            )
            val_loss = evaluate(
                model,
                val_data,
                micro_batch,
                block_size,
                eval_rng,
                device,
                batches=20 if args.smoke else 100,
                accelerator=accelerator,
                data_mode=data_mode,
            )
            last_val_loss = val_loss
            ppl = math.exp(min(20, val_loss))
            if accelerator.is_main_process:
                print(f"eval loss={val_loss:.4f} ppl={ppl:.2f}")
                metrics_logger.log_metrics(
                    step,
                    {
                        "eval/loss": val_loss,
                        "eval/ppl": ppl,
                    },
                )
                run_observer.metrics(
                    step,
                    {
                        "eval/loss": val_loss,
                        "eval/ppl": ppl,
                    },
                )
                run_observer.evaluation(step, val_loss, ppl)
            if data_mode == "indexed" and source_eval_cfg.get("enabled", True):
                source_batches = int(
                    source_eval_cfg.get(
                        "batches",
                        2 if args.smoke else 20,
                    )
                )
                source_results = evaluate_indexed_by_source(
                    model,
                    val_data,
                    eval_rng,
                    max(1, source_batches),
                    accelerator,
                )
                if accelerator.is_main_process:
                    source_metrics = {}
                    for source_id, result in source_results.items():
                        source_metrics[
                            f"eval/source/{source_id}/loss"
                        ] = result["loss"]
                        source_metrics[
                            f"eval/source/{source_id}/ppl"
                        ] = result["perplexity"]
                        print(
                            f"eval source={source_id} "
                            f"loss={result['loss']:.4f} "
                            f"ppl={result['perplexity']:.2f} "
                            f"tokens={result['tokens']:,}"
                        )
                    metrics_logger.log_metrics(step, source_metrics)
                    run_observer.metrics(step, source_metrics)
                    _append_jsonl(
                        os.path.join(output_dir, "source_eval.jsonl"),
                        {
                            "schema_version": 1,
                            "step": step,
                            "validation_corpus_sha256": val_data.corpus_sha256,
                            "sources": source_results,
                        },
                    )
            run_fixed_prompt_sample(
                model,
                accelerator,
                fixed_prompt_cfg,
                fixed_prompt_sampler,
                step,
                device,
                logger=metrics_logger,
            )

        if step % optim_cfg["save_interval"] == 0 or save_requested:
            ckpt_dir_name = f"step_{step:07d}"
            ckpt_dir = _make_local_checkpoint_dir(output_dir, ckpt_dir_name, local_checkpoint_mode)
            accelerator.wait_for_everyone()
            accelerator.save_state(ckpt_dir)
            save_training_progress(
                ckpt_dir,
                completed_steps,
                rng,
                process_index=accelerator.process_index,
                compatibility_fingerprint=compatibility_fingerprint,
                counters={"supervised_tokens": supervised_tokens_processed},
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, "model.pt"))

                should_upload = bool(
                    checkpoint_uploader
                    and checkpoint_uploader.enabled
                    and (
                        local_checkpoint_mode == "ephemeral"
                        or checkpoint_uploader.should_upload(step)
                    )
                )
                uploaded_ok = False
                if should_upload:
                    uploaded_ok = checkpoint_uploader.upload(ckpt_dir, step)

                manifest["last"] = ckpt_dir_name
                manifest.setdefault("steps", {})[ckpt_dir_name] = {
                    "step": step,
                    "val_loss": last_val_loss,
                    "timestamp": time.time(),
                    "uploaded": bool(uploaded_ok),
                    "local_path": ckpt_dir_name if local_checkpoint_mode == "persistent" else None,
                }
                if best_slots > 0:
                    _update_best_slots(manifest, ckpt_dir_name, last_val_loss, best_slots)
                _save_checkpoint_manifest(output_dir, manifest)
                run_observer.checkpoint(
                    step,
                    ckpt_dir_name,
                    uploaded=bool(uploaded_ok),
                    local_path=(
                        ckpt_dir_name
                        if local_checkpoint_mode == "persistent"
                        else None
                    ),
                )

                if should_upload and uploaded_ok:
                    checkpoint_uploader.prune_remote(manifest)

                if local_checkpoint_mode == "ephemeral":
                    if should_upload and uploaded_ok:
                        _cleanup_dir(ckpt_dir)
                    else:
                        persisted = _persist_ephemeral_checkpoint_dir(
                            ckpt_dir, output_dir, ckpt_dir_name
                        )
                        raise SystemExit(
                            "HF checkpoint upload failed in ephemeral mode. "
                            f"Saved local recovery checkpoint at: {persisted}"
                        )
                else:
                    if should_upload and not uploaded_ok:
                        print(
                            "[checkpoint_upload] Warning: upload failed, keeping local "
                            f"checkpoint at {ckpt_dir}"
                        )
                    rotate_checkpoints(
                        output_dir,
                        optim_cfg["checkpoint_limit"],
                        protected=_protected_steps(manifest),
                    )

        local_budget_stop = False
        if (
            observability_cfg.get("enabled", False)
            and step % run_observer.heartbeat_interval_steps == 0
        ):
            local_budget_stop = run_observer.heartbeat(
                step=step,
                tokens_processed=tokens_processed,
                force=True,
            )
            budget_flag = torch.tensor(
                1 if local_budget_stop else 0,
                device=device,
                dtype=torch.int32,
            )
            budget_flag = accelerator.reduce(budget_flag, reduction="sum")
            if int(budget_flag.item()) > 0:
                stop_training = True
                stop_reason = "runtime budget limit reached"
                save_requested = True
                if accelerator.is_main_process:
                    metrics_logger.maybe_print(
                        "[budget] Runtime cost limit reached; stopping after "
                        "a recovery checkpoint."
                    )

        if stop_training:
            break

    accelerator.wait_for_everyone()
    final_dir = os.path.join(output_dir, "final")
    if local_checkpoint_mode == "ephemeral" and not keep_local_final:
        final_dir = _make_local_checkpoint_dir(output_dir, "final", local_checkpoint_mode)
    accelerator.save_state(final_dir)
    save_training_progress(
        final_dir,
        completed_steps,
        rng,
        process_index=accelerator.process_index,
        compatibility_fingerprint=compatibility_fingerprint,
        counters={"supervised_tokens": supervised_tokens_processed},
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "model.pt"))
        print(f"Saved final checkpoint to {final_dir}")

        final_uploaded = False
        if checkpoint_uploader and checkpoint_uploader.enabled:
            final_uploaded = checkpoint_uploader.upload(
                final_dir,
                completed_steps,
                is_final=True,
            )
            if final_uploaded:
                checkpoint_uploader.prune_remote(manifest)

        manifest["final"] = {
            "timestamp": time.time(),
            "uploaded": bool(final_uploaded),
            "local_path": (
                "final"
                if os.path.abspath(final_dir) == os.path.abspath(os.path.join(output_dir, "final"))
                else None
            ),
        }
        _save_checkpoint_manifest(output_dir, manifest)
        run_observer.checkpoint(
            completed_steps,
            "final",
            uploaded=bool(final_uploaded),
            local_path=(
                "final"
                if os.path.abspath(final_dir)
                == os.path.abspath(os.path.join(output_dir, "final"))
                else None
            ),
        )
        if checkpoint_uploader and checkpoint_uploader.enabled:
            if artifact_upload_files:
                checkpoint_uploader.upload_artifacts(artifact_upload_files)
            checkpoint_uploader.upload_logs(output_dir)

        if local_checkpoint_mode == "ephemeral" and not keep_local_final:
            if final_uploaded:
                _cleanup_dir(final_dir)
            else:
                persisted = _persist_ephemeral_checkpoint_dir(final_dir, output_dir, "final")
                raise SystemExit(
                    "Final checkpoint upload failed in ephemeral mode. "
                    f"Saved local recovery checkpoint at: {persisted}"
                )
        elif not keep_local_final and final_uploaded:
            _cleanup_dir(final_dir)
        elif checkpoint_uploader and checkpoint_uploader.enabled and not final_uploaded:
            raise SystemExit(
                "Final checkpoint upload failed. "
                f"Local final checkpoint remains at: {final_dir}"
            )

        if stop_reason == "runtime budget limit reached":
            run_observer.terminal("budget_stopped", reason=stop_reason)
        elif stop_reason:
            run_observer.terminal("signal_stopped", reason=stop_reason)
        else:
            run_observer.terminal("completed")

        # Stop streaming data loader if active
        if streaming_mode:
            train_data.stop()

        cleanup_logging()

    if stop_reason:
        raise SystemExit(f"Training stopped safely: {stop_reason}")


if __name__ == "__main__":
    main()
