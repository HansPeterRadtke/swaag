#!/usr/bin/env python3
"""Offline refusal and file-safety checks for the source-profile renderer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import render_voice_sources as renderer


class SourceRendererTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="voice-source-renderer-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        original = Path(renderer.__file__).resolve().parent
        shutil.copytree(original / "templates", self.bundle / "templates")
        shutil.copy2(original / "experiment.json", self.bundle / "experiment.json")
        self.profile_value = json.loads((original / "deployment.example.json").read_text())
        self.profile = self.root / "profile.json"
        self.write_profile(self.profile_value)
        self.output = self.root / "rendered"
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "subprocess.Popen"):
            guard = patch(target, side_effect=AssertionError("Offline renderer attempted network or execution"))
            guard.start()
            self.addCleanup(guard.stop)

    def write_profile(self, value):
        self.profile.write_text(json.dumps(value))

    def change_manifest(self, edit):
        path = self.bundle / "experiment.json"
        value = json.loads(path.read_text())
        edit(value)
        path.write_text(json.dumps(value))

    def change_template(self, edit):
        manifest = json.loads((self.bundle / "experiment.json").read_text())
        spec = manifest["files"][0]
        path = self.bundle / spec["template"]
        path.write_bytes(edit(path.read_bytes()))
        spec["template_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.bundle / "experiment.json").write_text(json.dumps(manifest))

    def refused(self, phrase, original=False):
        with self.assertRaisesRegex(ValueError, phrase):
            renderer.render(self.bundle, self.profile, self.output, original)
        self.assertFalse(self.output.exists())

    def test_example_profile_renders_private_files_without_execution(self):
        receipt = renderer.render(self.bundle, self.profile, self.output)
        self.assertFalse(receipt["runtime_executed"])
        self.assertFalse(receipt["original_hashes_match"])
        self.assertEqual(renderer.OUTPUT_NAMES | {"_render-receipt.json"},
                         {p.name for p in self.output.iterdir()})
        self.assertEqual(0, self.output.stat().st_mode & 0o077)
        for name in renderer.OUTPUT_NAMES:
            self.assertEqual(0, (self.output / name).stat().st_mode & 0o077)
            self.assertNotIn(b"@@", (self.output / name).read_bytes())

    def test_original_hash_gate_rejects_different_deployment_before_writes(self):
        self.refused("Original source hash mismatch", original=True)

    def test_missing_and_unknown_profile_fields_are_refused(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                value = dict(self.profile_value)
                if unknown:
                    value["UNREVIEWED"] = "value"
                else:
                    value.pop("ASR_URL")
                self.write_profile(value)
                self.refused("Unknown or missing deployment profile fields")

    def test_duplicate_json_bindings_are_refused(self):
        raw = json.dumps(self.profile_value)
        self.profile.write_text(raw[:-1] + ', "MODEL_BASE_URL": "http://127.0.0.1:9"}')
        self.refused("Duplicate JSON key")

    def test_literal_injection_and_unsafe_addresses_are_refused(self):
        cases = [
            ("BROKER_OWNER", 'bad"literal'),
            ("BROKER_OWNER", "bad$(id)"),
            ("BROKER_OWNER", "bad\\escape"),
            ("BROKER_OWNER", "bad\nline"),
            ("BROKER_OWNER", "@@ASR_URL@@"),
            ("AGENT_PYTHON", "/tmp/../python"),
            ("EVIDENCE_PREFIX", "/"),
            ("EVIDENCE_PREFIX", "/tmp/no-trailing"),
            ("ASR_URL", "file:///tmp/audio"),
            ("ASR_URL", "http://user:secret@localhost:9/path"),
            ("ASR_URL", "http://localhost:9/path?token=value"),
            ("ASR_URL", "http://localhost:9/path#fragment"),
            ("ASR_URL", "http://localhost:65536/path"),
            ("ASR_URL", "http://localhost:0/path"),
            ("ASR_URL", "http://localhost:9/../path"),
            ("MODEL_BASE_URL", "http://localhost:9/path"),
            ("BROKER_HOST", "-bad-host"),
            ("SPEECH_LEASE_HOST", self.profile_value["RUNNER_LEASE_HOST"]),
        ]
        for key, replacement in cases:
            with self.subTest(key=key, replacement=replacement):
                value = dict(self.profile_value)
                value[key] = replacement
                self.write_profile(value)
                with self.assertRaises(ValueError):
                    renderer.render(self.bundle, self.profile, self.output)
                self.assertFalse(self.output.exists())

    def test_unknown_and_missing_template_markers_are_refused(self):
        self.change_template(lambda raw: raw.replace(b"@@MODEL_BASE_URL@@", b"@@UNREVIEWED_HOST@@"))
        self.refused("Unknown or missing template placeholders")

    def test_missing_template_marker_is_refused(self):
        self.change_template(lambda raw: raw.replace(b"@@MODEL_BASE_URL@@", b"static-value"))
        self.refused("Unknown or missing template placeholders")

    def test_malformed_marker_is_refused(self):
        self.change_template(lambda raw: raw + b"\n# @@malformed@@\n")
        self.refused("Malformed template placeholder")

    def test_template_hash_drift_is_refused(self):
        path = self.bundle / "templates" / "agent_rpc_worker.py.in"
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
        self.refused("Template content hash mismatch")

    def test_manifest_paths_cannot_escape_bundle_or_output(self):
        self.change_manifest(lambda value: value["files"][0].__setitem__("template", "../unrelated.py"))
        self.refused("Unapproved template path")

    def test_unknown_output_filename_is_refused(self):
        self.change_manifest(lambda value: value["files"][0].__setitem__("output", "../unrelated.py"))
        self.refused("Unknown or duplicate output files")

    def test_symlinked_template_is_refused(self):
        path = self.bundle / "templates" / "agent_rpc_worker.py.in"
        target = self.root / "unrelated.py"
        target.write_text("raise RuntimeError('must not execute')\n")
        path.unlink()
        path.symlink_to(target)
        self.refused("regular nonsymlink file")

    def test_existing_output_remains_unchanged(self):
        self.output.mkdir()
        sentinel = self.output / "valuable.txt"
        sentinel.write_text("retain this file")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            renderer.render(self.bundle, self.profile, self.output)
        self.assertEqual("retain this file", sentinel.read_text())
        self.assertEqual(["valuable.txt"], sorted(p.name for p in self.output.iterdir()))

    def test_symlinked_output_remains_unchanged(self):
        target = self.root / "existing"
        target.mkdir()
        sentinel = target / "valuable.txt"
        sentinel.write_text("retain this file")
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            renderer.render(self.bundle, self.profile, self.output)
        self.assertEqual("retain this file", sentinel.read_text())
        self.assertTrue(self.output.is_symlink())


if __name__ == "__main__":
    unittest.main()
