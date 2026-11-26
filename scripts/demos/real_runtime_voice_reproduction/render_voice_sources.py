#!/usr/bin/env python3
"""Render reviewed source templates offline; never run the voice runtime.

The profile supplies literal deployment values, not Python or shell fragments.
The output is a fresh private directory. Original-hash mode is an archival
source reproduction check, not a new live demonstration.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib
from urllib.parse import urlsplit

FORMAT = "swaag.voice-source-bundle.v1"
TOKEN = re.compile(rb"@@([A-Z][A-Z0-9_]*)@@")
HEX = re.compile(r"[0-9a-f]{64}")
FIELD_TYPES = {
    "MODEL_BASE_URL": "base_url",
    "ASR_URL": "url",
    "TTS_URL": "url",
    "BROKER_OWNER": "identifier",
    "BROKER_HOST": "hostname",
    "BROKER_EXECUTABLE": "path",
    "RUNNER_LEASE_HOST": "identifier",
    "SPEECH_LEASE_HOST": "identifier",
    "AGENT_PYTHON": "path",
    "AGENT_SITE_PACKAGES": "path",
    "SQLITE_PRELOAD": "path",
    "EVIDENCE_PREFIX": "directory_prefix",
}
OUTPUT_NAMES = {
    "verify_real_runtime_voice_once.py",
    "agent_rpc_worker.py",
    "verify_real_runtime_voice_evidence.py",
    "test_real_runtime_voice_evidence.py",
    "agent-config.template.toml",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_regular(path, limit=1_000_000):
    require(path.is_file() and not path.is_symlink(), "Expected a regular nonsymlink file: " + path.name)
    require(path.stat().st_size <= limit, "Input file exceeds size limit: " + path.name)
    return path.read_bytes()


def read_json(path, limit=1_000_000):
    raw = read_regular(path, limit)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs), raw


def validate_profile(profile):
    require(isinstance(profile, dict), "Deployment profile must be a JSON object")
    require(set(profile) == set(FIELD_TYPES), "Unknown or missing deployment profile fields")
    for key, kind in FIELD_TYPES.items():
        value = profile[key]
        require(isinstance(value, str) and 0 < len(value) <= 512, "Invalid deployment value: " + key)
        # These are literal contents already surrounded by source quotes.
        # Quotes, escapes, whitespace, substitutions and non-ASCII are refused.
        require(all(33 <= ord(char) <= 126 for char in value)
                and not any(char in value for char in "'\"\\`$@"),
                "Unsafe literal content: " + key)
        if kind in {"url", "base_url"}:
            require(re.fullmatch(r"https?://[A-Za-z0-9.-]+:[0-9]{1,5}(?:/[A-Za-z0-9._~/-]*)?", value),
                    "Invalid endpoint literal: " + key)
            parsed = urlsplit(value)
            try:
                port = parsed.port
            except ValueError:
                raise ValueError("Invalid endpoint port: " + key) from None
            require(parsed.hostname and port is not None and 1 <= port <= 65535
                    and not parsed.username and not parsed.password
                    and not parsed.query and not parsed.fragment,
                    "Invalid endpoint components: " + key)
            require(".." not in parsed.path.split("/"), "Endpoint path traversal: " + key)
            if kind == "base_url":
                require(parsed.path == "", "Model base must not contain a path")
        elif kind in {"path", "directory_prefix"}:
            core = value[:-1] if kind == "directory_prefix" and value.endswith("/") else value
            require(re.fullmatch(r"/[A-Za-z0-9._/+\-]+", core)
                    and not core.startswith("//")
                    and all(part not in {".", ".."} for part in core.split("/"))
                    and str(PurePosixPath(core)) == core,
                    "Invalid absolute deployment path: " + key)
            if kind == "directory_prefix":
                require(value.endswith("/") and core != "/", "Evidence prefix needs a non-root directory")
            else:
                require(not value.endswith("/"), "File/directory path must have no trailing slash: " + key)
        elif kind == "hostname":
            require(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value),
                    "Invalid broker host")
        else:
            require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value),
                    "Invalid broker identifier: " + key)
    require(profile["RUNNER_LEASE_HOST"] != profile["SPEECH_LEASE_HOST"],
            "The two expected resource hosts must differ")
    return profile


def prepare(bundle, profile_path, require_original=False):
    manifest, manifest_raw = read_json(bundle / "experiment.json")
    require(isinstance(manifest, dict)
            and set(manifest) == {"format", "experiment", "profile_fields", "files"}
            and manifest["format"] == FORMAT, "Unsupported source bundle manifest")
    require(manifest["profile_fields"] == FIELD_TYPES, "Profile schema does not match this renderer")
    profile, profile_raw = read_json(profile_path, 16_384)
    validate_profile(profile)
    specs = manifest["files"]
    require(isinstance(specs, list) and len(specs) == len(OUTPUT_NAMES), "Wrong output file set")
    names = [item.get("output") for item in specs if isinstance(item, dict)]
    require(len(names) == len(specs) and set(names) == OUTPUT_NAMES, "Unknown or duplicate output files")
    template_root = bundle / "templates"
    require(template_root.is_dir() and not template_root.is_symlink(), "Invalid template directory")
    outputs = {}
    observed_fields = set()
    for item in specs:
        require(set(item) == {"template", "output", "original_sha256", "template_sha256", "placeholders"},
                "Unknown or missing template manifest fields")
        name = item["output"]
        require(item["template"] == "templates/" + name + ".in", "Unapproved template path")
        for field in ("original_sha256", "template_sha256"):
            require(isinstance(item[field], str) and HEX.fullmatch(item[field]), "Invalid source digest")
        counts = item["placeholders"]
        require(isinstance(counts, dict) and set(counts) <= set(FIELD_TYPES)
                and all(type(count) is int and count > 0 for count in counts.values()),
                "Unapproved placeholder declaration")
        raw = read_regular(bundle / item["template"])
        require(sha(raw) == item["template_sha256"], "Template content hash mismatch: " + name)
        seen = Counter(match.decode("ascii") for match in TOKEN.findall(raw))
        require(dict(seen) == counts, "Unknown or missing template placeholders: " + name)
        require(b"@@" not in TOKEN.sub(b"", raw), "Malformed template placeholder: " + name)
        observed_fields.update(seen)
        rendered = raw
        for key in seen:
            rendered = rendered.replace(("@@" + key + "@@").encode("ascii"), profile[key].encode("ascii"))
        require(b"@@" not in rendered, "Unresolved source placeholder: " + name)
        if name.endswith(".py"):
            ast.parse(rendered, filename=name, feature_version=(3, 11))
        else:
            tomllib.loads(rendered.decode("utf-8"))
        if require_original:
            require(sha(rendered) == item["original_sha256"], "Original source hash mismatch: " + name)
        outputs[name] = rendered
    require(observed_fields == set(FIELD_TYPES), "An expected profile field is not used")
    matches = all(sha(outputs[item["output"]]) == item["original_sha256"] for item in specs)
    receipt = {
        "status": "rendered_offline_exact_original_sources" if matches else "rendered_offline_new_profile",
        "original_hashes_match": matches,
        "required_original_hashes": require_original,
        "manifest_sha256": sha(manifest_raw),
        "profile_sha256": sha(profile_raw),
        "rendered_sha256": {name: sha(raw) for name, raw in sorted(outputs.items())},
        "runtime_executed": False,
        "limitation": "Rendering and syntax checks do not establish live execution or new deployment qualification.",
    }
    return outputs, receipt


def render(bundle, profile_path, output, require_original=False):
    outputs, receipt = prepare(bundle, profile_path, require_original)
    output = output.absolute()
    require(not output.exists() and not output.is_symlink(), "Output already exists; refusing overwrite")
    require(output.parent.is_dir() and output.parent.resolve() == output.parent,
            "Output parent must be an existing canonical directory")
    output.mkdir(mode=0o700)
    created = []
    try:
        content = dict(outputs)
        content["_render-receipt.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
        for name, raw in content.items():
            destination = output / name
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            created.append(destination)
            with os.fdopen(fd, "wb") as sink:
                sink.write(raw)
                sink.flush()
                os.fsync(sink.fileno())
    except BaseException:
        # Only files created by this call are candidates for cleanup. Unknown
        # contents cause rmdir to fail and remain intact.
        for destination in reversed(created):
            destination.unlink()
        output.rmdir()
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-original-hashes", action="store_true")
    args = parser.parse_args()
    try:
        receipt = render(Path(__file__).resolve().parent, args.profile, args.output,
                         args.require_original_hashes)
    except (ValueError, OSError, SyntaxError) as exc:
        print(json.dumps({"status": "render_refused", "error_type": type(exc).__name__,
                          "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
