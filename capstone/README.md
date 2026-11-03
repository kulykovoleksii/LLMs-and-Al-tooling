# Personal Knowledge Assistant — classifier

The capstone currently has a deterministic offline baseline and an optional
classifier backed by a locally hosted Ollama model. Unsupported or mixed
requests should be marked `unclear` and require clarification. Neither path can
cause a side effect.

## Run

From the repository root:

```bash
python3 -m capstone "Що таке retrieval?"
python3 -m unittest discover -s capstone/tests -v
python3 -m capstone.evaluate
```

The default provider is deterministic and works offline. To try Ollama, start
the local Ollama service and provide a model that is already installed:

```bash
ollama list
python3 -m capstone --provider ollama --model codellama:7b-code "Що таке retrieval?"
```

You can set `OLLAMA_MODEL` instead of passing `--model`; `OLLAMA_URL` defaults
to `http://127.0.0.1:11434`. The request timeout defaults to 30 seconds and can
be changed with `--timeout`. No API key, Python package, or paid service is
required. Ollama runs the selected model locally, but downloading a model may
use network bandwidth and disk space. Local model output is still a prediction;
evaluate it before relying on its classifications.

The CLI prints a JSON object with exactly these fields:

```json
{
  "intent": "knowledge_query",
  "needs_clarification": false,
  "reason": "matched_knowledge_rule"
}
```

Possible intents are `knowledge_query`, `draft_request`, and `unclear`.
`unclear` always requires clarification. This is a deliberately small keyword
baseline, not a general language classifier; `data/golden.jsonl` is a synthetic
smoke-test set, not evidence of production accuracy.

## Current acceptance checks

- The output is validated against a strict, typed schema.
- Blank input and malformed output are rejected rather than silently accepted.
- Mixed or unrecognized requests require clarification.
- The golden set covers English and Ukrainian examples and runs without an API
  key or network access.
- The Ollama adapter requests constrained JSON output and validates it using
  the same classification schema; HTTP and malformed-output errors are surfaced.
- `python3 -m capstone.evaluate --provider ollama --model MODEL` compares the
  local model to the fixed synthetic set and reports accuracy and mismatches.

The evaluation reports a measurement, not a quality guarantee or automatic
pass/fail decision. The rules provider remains the default until a model earns
that choice on an appropriately representative evaluation set. In the initial
run, the installed `codellama:7b-code` model achieved 7/26 exact matches
(27%) on this small set, so it is not suitable for this classifier as-is.
