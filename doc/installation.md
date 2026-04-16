# SWAAG installation

## Repository checkout

```bash
cd /data/src/github
 git clone https://github.com/HansPeterRadtke/swaag.git
cd swaag
```

## Python install

Editable developer install:

```bash
python3 -m pip install -e .[test]
```

Benchmark extras:

```bash
python3 -m pip install -e .[official-benchmarks]
```

Packaging / publishing extras:

```bash
python3 -m pip install -e .[publish]
```

If the active environment is not writable for publish tooling, build in a
throwaway venv instead:

```bash
python3 -m venv /tmp/swaag-build
/tmp/swaag-build/bin/python -m pip install build twine
/tmp/swaag-build/bin/python -m build
```

## `llama.cpp`

Official project:
- https://github.com/ggml-org/llama.cpp

SWAAG expects the `llama.cpp` HTTP server endpoints `/health`, `/tokenize`, and `/completion`.

A practical general-purpose local model:
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF

Example launch:

```bash
llama-server \
  -m /absolute/path/to/Qwen2.5-7B-Instruct-Q5_K_M.gguf \
  --host 127.0.0.1 \
  --port 14829 \
  -c 2048
```

## Config override paths

Override precedence:
1. `src/swaag/assets/defaults.toml`
2. `config/local.toml`
3. `SWAAG_CONFIG=/path/to/file.toml`
4. `SWAAG__SECTION__KEY=value`

Example:

```bash
export SWAAG__MODEL__BASE_URL=http://127.0.0.1:14829
export SWAAG__MODEL__CONTEXT_LIMIT=2048
export SWAAG__SESSIONS__ROOT=/tmp/swaag-sessions
```

## First verification steps

```bash
python3 -m swaag doctor --json
python3 -m swaag.tests
pytest -q tests/test_config.py tests/test_cli.py -x
```
