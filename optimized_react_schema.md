# Optimized React JSON Schema (super_bench/runs/progress/program_8/optimized_react.json)

This document summarizes the structure of the optimized React run logs and clarifies how the `prediction` field is represented.

**File Shape**
- Root: JSON array with 27 items.
- Each item has top‑level keys: `example`, `prediction`, `score`.

**example**
- `instance_id` (string): Unique id for the task instance.
- `github_repo` (string URL): Target repository.
- `git_commit` (string SHA): Commit to check out.
- `query` (string): Full natural‑language task.
- `query_components` (object): Parsed query parts
  - `e2e_task` (string)
  - `scenario_task` (string)
  - `report` (string)
  - `instructions` (string)
- `solution_dependencies` (string): Space‑separated `pip` style requirements (often long).
- `answer` (string): Ground‑truth answer as a JSON string literal (e.g., '{"Sentence-level BLEU": 0.0, "Document-level BLEU": 0.01}').
- `landmarks` (array<string>): Regexes or log snippets expected during a correct run.
- `solution` (array<object>): Reference steps the solver should execute
  - Each element has:
    - `action` (object): `{ type: "execute" | ..., content: string }`
    - `observation` (string): Expected console output or notes (may be empty).

**prediction**
- Represents the model’s attempt and outcome for this example.
- Keys: `trajectory`, `reasoning`, `result`.

- `trajectory` (object): Flattened, indexed sequence of tool‑use steps using the pattern below. Index starts at 0 and increases by 1 for each step.
  - `thought_{i}` (string): The agent’s internal reasoning at step `i`.
  - `tool_name_{i}` (string): Tool invoked at step `i` (observed values include `run_command`, `read_file`, `edit_file`, `write_file`, `finish`).
  - `tool_args_{i}` (object): Arguments passed to the tool at step `i`.
    - Common keys seen: `command`, `filename`, `file_path`, `edits`, `path`, `content`, `lines_start`, `lines_end`, `start_line`, `end_line`.
  - `observation_{i}` (object|string): Result of the tool call.
    - Common object keys: `success` (bool), `output` (string), `message` (string), optionally `error`, `file_path`, `content`, and line range metadata.
- `reasoning` (string): A compact, natural‑language summary explaining what was attempted, why, and any blocking issues.
- `result` (string): A single summary line that encodes
  - `success` flag (e.g., `success=False`),
  - `structured_output` dictionary literal when applicable (e.g., `{'Sentence-level BLEU': 0.0, 'Document-level BLEU': 0.0}`),
  - and an additional free‑text `reasoning` segment. Note this field is a textual summary, not a nested JSON object.

**score**
- `score` (number): Overall scalar score for the item.
- `score_dict` (object): Subscores
  - Observed keys: `submitted` (0/1), `output_match` (number), `landmarks` (number).

**Notes & Conventions**
- The log is optimized for replay/analysis rather than strict JSON normalization of every subfield. In particular, `example.answer` stores JSON as a string, and `prediction.result` stores a textual encoding of structured and unstructured data.
- The `trajectory` indexing allows simple chronological reconstruction without nesting arrays; consumers should group fields by shared suffix index.
- Not all `observation_{i}` values are objects—some steps capture errors or raw strings.

**Quick Stats (sampled across file)**
- Items: 27
- Typical `trajectory` length: ~5–100 thoughts/steps per item.
- Common tools: `run_command` (dominant), then `read_file`, `edit_file`, `write_file`, `finish`.

**Example (abridged)**
```jsonc
{
  "prediction": {
    "trajectory": {
      "thought_0": "List files; locate dataset/scripts.",
      "tool_name_0": "run_command",
      "tool_args_0": { "command": "ls -lR" },
      "observation_0": { "success": true, "output": "Exit code: 0..." },
      "thought_1": "Clone repo and checkout commit.",
      "tool_name_1": "run_command",
      "tool_args_1": { "command": "git clone ... && git checkout <sha>" },
      "observation_1": { "success": true, "message": "executed in 1.2sec" }
      // ...
    },
    "reasoning": "Summary of what was attempted and why...",
    "result": "success=False structured_output={'Sentence-level BLEU': 0.0, 'Document-level BLEU': 0.0} reasoning='...why it failed...'"
  }
}
```

**Consumer Guidance**
- To iterate steps, scan `trajectory` keys by numeric suffix and group quadruples `(thought_i, tool_name_i, tool_args_i, observation_i)`.
- Parse `prediction.result` as text if you need the `success` flag or the embedded `structured_output`; it is not guaranteed to be valid JSON.
- Treat `example.answer` as JSON-encoded text; parse it before numeric comparisons.

