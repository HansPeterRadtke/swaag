# Test execution commands

Authoritative categories:

| Category | Command | Purpose |
| --- | --- | --- |
| code_correctness | `python3 -m swaag.testprofile code-correctness` | deterministic code checks |
| agent_test | `python3 -m swaag.testprofile agent-tests` | cached agent behavior checks including the full cached benchmark catalog |
| combined | `python3 -m swaag.testprofile combined` | code_correctness first, then agent_test only if green |

Artifact-producing combined command:

```bash
python3 -m swaag.benchmark test-categories --clean --output /tmp/swaag-test-categories
```

Manual validation / real usage, not tests:

```bash
python3 -m swaag.benchmark manual-validation --clean --validation-subset --output /tmp/swaag-manual-validation
```
