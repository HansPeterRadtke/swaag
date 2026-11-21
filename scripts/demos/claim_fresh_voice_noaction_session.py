#!/usr/bin/env python3
"""Single-use, fail-closed admission for a new, PRIVATE zero-tool SWAAG voice session.

Invoked on Nitro by the Jetson acceptance harness immediately before a real
model request. A claim is a local test marker, not authorization to run tools.
Its creation is deliberately non-idempotent: any failed or uncertain attempt
must be investigated, not blindly repeated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from datetime import datetime,timezone


SESSION_ROOT_PATTERN = re.compile(r"/data/var/swaag-benchmarks/[A-Za-z0-9][A-Za-z0-9_-]{0,95}/sessions")
CONFIG_PATTERN = re.compile(r"/data/var/swaag-benchmarks/[A-Za-z0-9][A-Za-z0-9_-]{0,95}/noaction-config\.toml")
CLAIM_NAME = ".one_use_voice_noaction_claim.json"


def validate_and_claim(config_path: Path, expected_sha256: str) -> dict:
    """Claim exactly one session root and reject any previous durable turn.

    Creates an exclusive mode-0600 record before inspecting the session tree,
    preventing concurrent harness invocations from sharing a model session.
    Never deletes an existing claim and never changes canonical user state.
    """
    if (type(expected_sha256) is not str or
            re.fullmatch(r"[0-9a-f]{64}",expected_sha256) is None):
        raise ValueError("Expected exact SHA-256 is required")
    supplied = str(config_path)
    if CONFIG_PATTERN.fullmatch(supplied) is None or config_path.is_symlink():
        raise ValueError("Refusing an untrusted or symlinked noaction configuration")
    if not config_path.is_file():
        raise FileNotFoundError("No private noaction configuration")
    data = config_path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_sha256:
        raise ValueError("Noaction config checksum differs from admitted source")
    cfg = tomllib.loads(data.decode("utf8"))
    tools = cfg.get("tools") or {}
    if (tools.get("enabled") != [] or tools.get("staged_discovery") is not False or
            tools.get("allow_stateful_tools") is not False or
            tools.get("allow_side_effect_tools") is not False):
        raise ValueError("Noaction source has unapproved system tool access")
    session_root = Path(str((cfg.get("sessions") or {}).get("root", "")))
    if (SESSION_ROOT_PATTERN.fullmatch(str(session_root)) is None or
            session_root != config_path.parent / "sessions" or
            session_root.is_symlink() or
            session_root.parent.resolve() != config_path.parent.resolve()):
        raise ValueError("Sessions root must be the matching private test folder")
    if not session_root.is_dir():
        raise FileNotFoundError("Private session directory was not precreated")
    # Atomically claim BEFORE checking pre-existing session artifacts.
    # O_EXCL makes concurrent attempts fail immediately and avoids TOCTOU
    # between two independent test harnesses.
    claim_file = session_root / CLAIM_NAME
    fd = os.open(claim_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        prior = sorted(p.name for p in session_root.iterdir() if p.name.startswith("session_"))
        result = {
            "claimed_at_utc":datetime.now(timezone.utc).isoformat(),
            "configuration_sha256":expected_sha256,
            "private_session_root":str(session_root),
            "prior_session_directories":prior,
            "one_use_only":True,
            "allowed_agent_tools":[],
            "admitted":not prior,
        }
        encoded=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
        os.write(fd,encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    if prior:
        raise ValueError("Existing source session present; one-use claim retained to prevent replay")
    return result


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--sha256",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(validate_and_claim(args.config,args.sha256),sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
