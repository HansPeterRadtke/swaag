# Exact source reproduction for the recorded-voice runtime demonstration

This bundle reconstructs the five source files used to run and independently
verify a bounded SWAAG voice exchange on 26 November 2026. Internal deployment
addresses, machine labels, interpreter paths and the private evidence prefix are
replaced by twelve reviewed literal placeholders. A separate operator-owned
JSON profile supplies those values. The profile, raw evidence and audio are
not included.

The renderer does not execute the agent, contact a service, install packages,
or modify an existing output directory. With the original private profile and
`--require-original-hashes`, all five rendered files must match the exact
already-executed source hashes in `experiment.json`. This is source
reconstruction, not another live test.

## Observed result and its scope

One 4.3-second public recording saying “Can you repeat that for me?” passed
through the actual voice WebSocket server, voice gateway, SWAAG
`AgentRuntime` and `CommunicationService`, the resident model and actual
speech synthesis. The recording was used in a fresh empty conversation.
The model answered: “I don't have a previous message to repeat. What would you
like me to say?” The original model output, agent response, gateway output
and durable conversation agreed.

The unsent draft survived reconnection. Forged client owner/speaker flags and
a forged automatic commit did not submit it; explicit test-only manual
confirmation was required. A separate operation canceled before handoff was
rejected by the actual agent when subsequently submitted, with no model
inference for that canceled operation. This did not interrupt a running model.

After disconnecting during answer audio, reconnection returned the same
128,000 generated PCM bytes, representing four seconds. A normal client
playback acknowledgement became durable. A fresh agent process then replayed
the same completed answer without model egress or a changed canonical journal.
There was one model completion and zero agent tool calls. The backend also
received metadata, tokenization and template requests; “one completion” does
not mean “one total HTTP request.”

| Measured operation | Seconds |
| --- | ---: |
| Speech-recognition request | 1.966 |
| Gateway answer operation | 25.799 |
| Model completion HTTP request, within the gateway operation | 16.120 |
| Speech-synthesis request | 3.107 |
| Entire isolated run, including setup, cancellation, reconnect, restart and cleanup | 53.900 |

These are overlapping measurements from one sample and must not be summed.
The relays buffer responses; these figures do not establish streaming
first-token latency, microphone-to-ear latency, percentiles or a service
guarantee. The independent audit passed, and fourteen deliberately corrupted
evidence cases were rejected without another inference request. Exact
unrounded measurements and evidence digests are recorded in `experiment.json`.

An earlier attempt failed with HTTP 500 before ASR, model or synthesis because
the installed WebSocket library's top-level server alias selected an
incompatible API. The corrected voice source explicitly selects the legacy
server API expected by its handler. The real listener regression reproduced
the old failure and passed after the correction on Python 3.12.3/websockets
15.0.1 and Python 3.8.10/websockets 13.1. The selected source regression group
passed 111 tests; this is not a whole-repository test claim.

This is recorded audio sent through the real network interface, not a live
microphone or physical phone test. A software playback acknowledgement does
not prove audible playback or audio quality. Installed production services
were not changed. Automatic turn detection, speaker authentication,
hands-free command authorization, lost-playback-acknowledgement recovery and
running-model cancellation remain outside this result. The worker omits
background server-startup loops. Its Python admission hooks are not an
operating-system sandbox. This no-tool conversational sample is distinct from
the [separate multi-step computer-task demonstration](../../../docs/evaluation/nov23-verified-firstpass-reconnect-provenance-agent-demo.md).

## Fixed sources, input and model configuration

The executed SWAAG runtime revision was
`06fcf206f2b448cb8f1f518377e38105b50cbd60`; the executed voice-runtime revision
in the separate DevTests repository was
`28ab9dc2ca166b9bdc942939a510ec8ebb7ee01a`. A later commit publishing this
bundle is not itself a new live runtime qualification.

The input comes from **LiveKit, LiveKit EoT Benchmark**,
`livekit/eot-bench-data`, revision
`ae834af9e9bddf73d1f550c190aaeb99bf31f523`, English validation split,
`general` cohort, row `en__489_62__user_turn_009`.
The [pinned publisher dataset card](https://huggingface.co/datasets/livekit/eot-bench-data/resolve/ae834af9e9bddf73d1f550c190aaeb99bf31f523/README.md)
declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [pinned source shard](https://huggingface.co/datasets/livekit/eot-bench-data/resolve/ae834af9e9bddf73d1f550c190aaeb99bf31f523/data/en/validation-00000-of-00002.parquet)
has SHA-256
`52bf7ff71e83d0214d744693839f8a6b14e78e0f26db20dfdb7f589453a50103`.
Its native row audio bytes were extracted unchanged; no resampling or
re-encoding was performed. The WAV SHA-256 is
`60fa1df444a16cdf558ca052cf087f49b9ed7ed238bf68d0b24e7906aaac71c0`:
68,800 frames, 16,000 Hz, mono PCM16. Prior dataset conversation context was
not supplied. This bundle redistributes no input or generated audio.

The observed local model profile was `qwen38-27b-q8-full-262k`, using
llama.cpp, temperature zero, top-p one, seed zero, thinking and cache disabled,
no retries and a 110-second generation ceiling. The actual request allowed
192 output tokens. The server reported model basename
`Qwen3.8-27B-UD-Q8_K_XL.gguf` and build `b9842-6f4f53f2b`.
Its properties-response SHA-256,
`da941b44ecf54a3843ae68ebf28b9e85ffa3f8978df98a044e43fab8b0726357`,
matches all five properties-response hashes recorded during the accepted run.
The profile label and reported basename do not establish a model-weight
digest or public model repository revision; neither was established here.

The original execution used Python 3.11.9 and SQLite 3.51.3 for the agent,
and Python 3.12.3 with websockets 15.0.1 for the voice/gateway process.
Both agent tool allowlists were empty. This bundle preserves the exact
configuration source, but does not provide an environment lockfile, repair
a different interpreter environment, or install any dependency.

The historical November 20 Python 3.11 syntax failure was reproduced from its
pre-fix source. The current pinned SWAAG revision runs in the qualified Python
3.11 environment; 118 selected SWAAG boundary tests passed. A separate native
Python 3.12 agent environment still lacks `opentelemetry.context` and has an
older, unsuitable SQLite version. Those environment gaps are distinct from the
repaired WebSocket API issue and its 111 selected voice tests. Neither selected
group represents the entire repository suite.

## Offline renderer

Use Python 3.11 or later. From this directory:

```sh
python3.11 -B -m unittest test_render_voice_sources -v
python3.11 -B render_voice_sources.py \
  --profile deployment.example.json \
  --output /private/new-example-render
```

The example contains inert loopback endpoints and illustrative paths. It
demonstrates source rendering only. Its receipt reports
`rendered_offline_new_profile`; it cannot inherit the historical live result.

For archival reconstruction with the original operator-held profile:

```sh
python3.11 -B render_voice_sources.py \
  --profile /private/deployment.json \
  --output /private/new-exact-render \
  --require-original-hashes
```

The output's parent must already exist and be canonical. The output must not
exist, including as a symlink. The renderer rejects duplicate JSON bindings,
unknown or missing fields, unsafe literal characters, undeclared/missing
placeholders, unexpected source hashes and output-name/path changes. It
parses the resulting Python and TOML without importing those programs.
It creates the output with mode 0700 and files with mode 0600.

The twelve fields are documented by `experiment.json` and
`deployment.example.json`. They contain URLs, broker identity/host labels,
absolute interpreter/library paths and the allowed evidence-directory
prefix. They are literal values, not shell fragments or environment-variable
expressions. Keep real profiles and rendered source outside published trees.
No original result file is rewritten.

## Runtime reproduction recipe

This section documents the already-executed harness interface. Running it
again is a new service request and needs current authorization and fresh
resource allowances. A successful render, preflight or publication does not
supply either. Do not rerun an existing attempt to recover missing results.

Supply these operator-owned dependencies:

- A clean SWAAG checkout at the pinned runtime revision.
- A local DevTests Git repository containing the pinned voice revision.
- Qualified agent and voice Python environments, as described above.
- The exact input WAV obtained from the attributed pinned dataset row.
- The operator's voice JSON template with SHA-256
  `03ad88af3b85511316a07198633bf0812bea513029466a4b4e28a8dc9a8222ca`.
  This private dependency is deliberately not bundled.
- The rendered five sources and the private deployment profile.
- Resident ASR, model and synthesis services matching the admitted deployment,
  and a broker whose fresh status grants the configured owner both expected
  machine allowances.

The exact runner expects the complete exported voice source at
`WORK_ROOT/source/voice_agent_live_runtime` and `baseline.json` in
`WORK_ROOT`. Keep that layout; the original source is reconstructed without
changing its path assumptions. Choose a new `WORK_ROOT` below the configured
`EVIDENCE_PREFIX`; its parent must already exist.

After setting the capitalized variables to operator-owned absolute paths,
prepare a fresh source export. The shell's fail-fast settings stop if the
work directory already exists:

```sh
set -euo pipefail
mkdir "$WORK_ROOT"
mkdir "$WORK_ROOT/source"
git -C "$VOICE_REPO" archive \
  28ab9dc2ca166b9bdc942939a510ec8ebb7ee01a voice_agent_live_runtime \
  | tar -x -C "$WORK_ROOT/source"
printf '%s\n' '{"swaag":"06fcf206f2b448cb8f1f518377e38105b50cbd60","devtests":"28ab9dc2ca166b9bdc942939a510ec8ebb7ee01a"}' \
  > "$WORK_ROOT/baseline.json"
VOICE_SOURCE="$WORK_ROOT/source/voice_agent_live_runtime"
EVIDENCE="$WORK_ROOT/attempt"
```

The default invocation is preflight only:

```sh
"$VOICE_PYTHON" -B "$RENDERED/verify_real_runtime_voice_once.py" \
  --swaag-source "$SWAAG_SOURCE" --voice-source "$VOICE_SOURCE" \
  --template "$VOICE_TEMPLATE" --fixture "$FIXTURE" \
  --worker "$RENDERED/agent_rpc_worker.py" \
  --config-template "$RENDERED/agent-config.template.toml" \
  --evidence "$EVIDENCE"
```

Only after independently obtaining current authorization and service/broker
admission, use the same command with the explicit
`SWAAG_W111_REAL_RUNTIME_ONCE=1` environment opt-in and `--run`.
The output directory must still be absent. The unchanged harness permits
one ASR request, at most one model completion and one synthesis request;
it retains failure evidence rather than retrying an uncertain operation.
A new deployment profile, dependency set, model instance or source revision
requires its own qualification; the historical measurements do not transfer.

## Independent evidence verification

Verification reads retained evidence and Git objects. It does not call
services or import the agent runtime. For historical evidence, use the
original audited SWAAG source location recorded in that evidence. Do not
rewrite recorded absolute paths or journals to fit a moved checkout.

```sh
"$VOICE_PYTHON" -B "$RENDERED/verify_real_runtime_voice_evidence.py" \
  --evidence "$EVIDENCE" \
  --runner "$RENDERED/verify_real_runtime_voice_once.py" \
  --worker "$RENDERED/agent_rpc_worker.py" \
  --voice-source "$VOICE_SOURCE" --swaag-source "$SWAAG_SOURCE" \
  --devtests-repo "$VOICE_REPO" \
  --expected-devtests-revision 28ab9dc2ca166b9bdc942939a510ec8ebb7ee01a \
  --conversation w111-real-swaag-confirmed-public-repeat-wscompat
```

The two explicit final arguments are required for this accepted attempt:
the frozen historical verifier retains earlier defaults for a preserved
failed attempt. The corruption driver uses the same arguments plus
`--verifier "$RENDERED/verify_real_runtime_voice_evidence.py"`:

```sh
"$VOICE_PYTHON" -B "$RENDERED/test_real_runtime_voice_evidence.py" \
  --evidence "$EVIDENCE" \
  --runner "$RENDERED/verify_real_runtime_voice_once.py" \
  --worker "$RENDERED/agent_rpc_worker.py" \
  --voice-source "$VOICE_SOURCE" --swaag-source "$SWAAG_SOURCE" \
  --devtests-repo "$VOICE_REPO" \
  --verifier "$RENDERED/verify_real_runtime_voice_evidence.py" \
  --expected-devtests-revision 28ab9dc2ca166b9bdc942939a510ec8ebb7ee01a \
  --conversation w111-real-swaag-confirmed-public-repeat-wscompat
```

The corruption driver changes only in-memory reads. A passing corruption
suite tests the verifier; it is not another voice demonstration. Keep raw
evidence, deployment profiles, and original service receipts private.
