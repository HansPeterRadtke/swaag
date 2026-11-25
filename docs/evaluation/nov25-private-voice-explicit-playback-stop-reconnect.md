# Explicit playback stop survives reconnect and restart — private candidate

**Observed result, November 25:** the earlier W71 guarded voice-server
candidate could automatically restart a reply after the client explicitly
stopped playback and immediately reconnected. This was reproduced using
a real private server process, actual WebSockets and a saved 5.44-second
previously generated public-test WAV. No playback-completed acknowledgement
was sent: the user stopped before the first stream ended. This is **not**
the previously denied physical lost-ACK experiment.

The new source-pinned candidate persists an acknowledged stop for the
specific assistant turn, stream and generation before replying to the
client. A later ready-state update cannot resurrect the stopped reply,
and a late completion update cannot mark deliberately stopped playback
as completed. The answer and WAV remain available for deliberate replay;
this does not cancel agent computation or authorize a user action.

## Real execution and independent verification

The previously verified W99 public-test answer WAV
(`f8da0c638b6771faac445d831cdae16cd4315ecd86db0a63fa22e1290666dc4f`)
was copied into a newly seeded synthetic private dialogue for each run.
The fixture dialogue text is synthetic; the media bytes are genuine
previous output. No model, speech recognition, TTS, phone, live agent or
external provider was invoked. All external model, worker and agent
routes were replaced with inert localhost endpoints. No production
service or user history was edited.

| Actual process test | Observed result |
| --- | --- |
| W71 unmodified base, stop after first 1,920 PCM bytes, reconnect before end | Incorrect new automatic audio start, new stream ID; no durable stop marker |
| New candidate, same separate test protocol | No automatic restart; one saved stop, completed answer, uncompleted playback |
| New candidate process with prior stopped state copied into isolated storage | No automatic audio; deliberate replay returned all 174,080 original PCM bytes |
| Separate ordinary pre-end disconnect, without stop | Entire 174,080 bytes recovered on new stream; normal acknowledgement persisted |

The last process also imported the actual candidate source and exercised
the real private storage functions: obsolete socket rejection, wrong
stream and generation rejection, preservation of an already recorded
genuine completion, refusal to re-admit a stopped stream, and rejection
of resurrection by a later ready or completion update. The three
process experiments finished in 0.855, 2.333 and 11.474 seconds
respectively; these are test scenario durations, not speech latency.

The independent read-only verifier
`scripts/demos/verify_private_voice_durable_stop_evidence.py` checks exact
as-run script and original result hashes, distinct stream IDs, event
ordering, private durable history and actual saved media fingerprints.
It returned
`verified_actual_private_stop_reconnect_fix_and_regressions`.
Separate synthetic unit tests challenge the verifier and source gate.

## Source and evidence

Source transformation:
`scripts/demos/prepare_private_voice_durable_audio_stop.py`.
It admits only the original seven-file private W71 source set, replaces
exact source anchors, parses the result and writes a new private candidate.
Original server SHA-256:
`873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf`.
Corrected private server SHA-256:
`cd7ccfa7019e4dd63a2ed25e01ba257f405ebaffe3a415364f1724e931b134ea`.

Actual private process runners:
`scripts/demos/verify_private_voice_stop_reconnect.py` and
`scripts/demos/verify_private_voice_durable_stop_followup.py`.
The original full source and all receipts are privately retained on Jetson:
`/data/var/swaag-benchmarks/owner-serial-000100-20261125/`.
See the `baseline-stop-before-end`, `corrected-stop-before-end`, and
`stop-restart-and-normal-recovery` subdirectories, including original
attempts, results, event and server logs, and durable histories.
Read-only revalidation, with no action or model replay:

```
python3 scripts/demos/verify_private_voice_durable_stop_evidence.py \
  --evidence /data/var/swaag-benchmarks/owner-serial-000100-20261125
```

This private source successor is not a deployed production repair.
The established production voice process was unchanged. The source-only
SWAAG repository commit makes the candidate transformation and test
reproducible but does not install it or merge unrelated DevTests/Android
workers' changes. The dedicated automatic-turn classifier remains
unavailable; physical Android microphone, audible playback, speaker
liveness, and automatic command authorization remain unverified. The
real fresh-synthesis stop path is distinct from the retained-WAV delivery
test and needs its own no-action regression.
