# Command reference

Every command of this build: what it does, its flags, the files it writes and its
exit codes. `werkbank --help` and `werkbank <command> --help` print the installed
build's flags. The short path is in the [README](../../README.md); how to read
what the commands write is in [reading your results](READING_RESULTS.md).

| Command | Section |
|---|---|
| `./werkbank`, `./install.sh`, `werkbank doctor`, `werkbank harbor-doctor`, `werkbank harbor-smoke` | [Installation](#installation) |
| `werkbank start` | [Guided start](#guided-start) |
| `werkbank run`, `werkbank list-cases` | [Run a model](#run-a-model) |
| `werkbank preflight` | [Check the server first](#check-the-server-first) |
| `werkbank validate`, `werkbank regrade`, `werkbank export` | [Validate, regrade and export](#validate-regrade-and-export) |
| `werkbank report`, `werkbank compare`, `werkbank result` | [Reports and evaluation](#reports-and-evaluation) |
| `werkbank submit` | [Submit results](#submit-results) |
| `werkbank-ui` | [Browser interface](#browser-interface) |

The examples run in the root of the checkout, where the installed commands are
`.venv/bin/werkbank` and `.venv/bin/werkbank-ui`.

## What the commands contact

- **Your model server:** only `werkbank start`, `werkbank preflight` and `werkbank
  run`, and only at the endpoint you give or the one server on this computer that
  `werkbank start` finds or you choose there. Requests ignore the host's proxy
  settings, never follow a redirect and are never retried. They carry no API key
  and no authorization header unless you set `WERKBANK_API_KEY`, and then only to
  a server you named ([API key](#api-key)).
- **The maintainer's receiver:** only the submit flow (`werkbank submit`, or
  `werkbank start` after its offer), and only after you saw what is sent and
  confirmed it ([submit results](#submit-results)).
- **Nothing else.** Every other command works offline. The installation needs the
  network and contacts no model. There is no telemetry and no paid judge.

## Installation

| Command | What it does |
|---|---|
| `./werkbank [start flags]` | Runs `./install.sh` on its first call, and whenever no complete installation is found, then `werkbank start` with every argument passed on unchanged. `./werkbank --help` describes both. |
| `./install.sh` | Checks the prerequisites, creates `.venv` from the hash-locked requirements, pulls the pinned container image and runs the two checks below. `--skip-docker` installs the Python environment only; `--check` changes and prints nothing and exits `0` only for a complete installation. |
| `werkbank doctor --offline` | Prints the installed version, the harness digest (`harness_sha256`, the field a run's `plan.json` records), Python and the platform as JSON. `--offline` is required. |
| `werkbank harbor-doctor [--image IMAGE]` | Read-only check that Harbor, Docker and the image are ready; prints its findings as JSON. |
| `werkbank harbor-smoke --output NEW [--image IMAGE]` | Runs one task against a built-in fake endpoint as a check of the runtime and writes its evidence and `smoke.json` into the new directory. No model is contacted or measured. |

`IMAGE` is a local `sha256:` image ID or a `repository@sha256:` digest, never a
tag; the default is the pinned runtime image. Requirements, every step, the
installation from a release archive or by hand, what a failed check means,
updating and removing: [installation](../INSTALLATION.md).

## Guided start

`werkbank start` takes you from a server on this computer to a
[result page](#result-page): it finds the server and the model, checks the server,
runs three passes, evaluates them and writes the result page. It contacts the
model. Every choice can be given as a flag, and with `--yes` nothing is asked.

```sh
./werkbank
./werkbank --browser
.venv/bin/werkbank start
.venv/bin/werkbank start --endpoint http://127.0.0.1:8000/v1 --model served-name \
  --mode quick --yes
```

| Mode | Runs | Passes | Result |
|---|---|---|---|
| Quick (`--mode quick`) | the case set `quick-v3`, 14 cases: the easy and medium ones that a 64K context can run | three with seeds 1, 2 and 3, or one with seed 1 | a result to compare, without an overall score; three passes can be submitted |
| Normal (`--mode normal`) | the case set `normal-v1`, 31 cases: every case that is not very hard | as Quick | a result to compare, with an overall score over its 31 cases; three passes can be submitted |
| Full (`--mode full`) | the case set `full-v1`, all 40 cases | as Quick | a result to compare, with an overall score over all 40 cases; three passes can be submitted |
| Trial (`--mode trial`) | seven easy cases, one per category: M04, S06, DA05, R03, A15, C01 and L01 | one with seed 1 | a check that the installation, the agent loop, the containers and the result page work; no benchmark, no score, not comparable, never submitted |

A single pass of Quick, Normal or Full takes about a third of the time and is never
submitted. Its result page says that it is a single pass; without temperature 0
it has no core number and no score
([why three passes](READING_RESULTS.md#why-three-passes)).

### Steps

1. **Readiness.** The read-only check of `harbor-doctor`, before any contact. When
   something is missing, start names the fix (`./install.sh`, or `docker pull
   IMAGE` for an image given with `--image`) and stops. It installs and pulls
   nothing.
2. **Server.** Without `--endpoint`, start sends one read-only `GET
   http://127.0.0.1:<port>/v1/models` to each usual port, in this order: 8000
   (vLLM), 8080 (llama.cpp), 30000 (SGLang), 11434 (Ollama), 1234 (LM Studio), 5000
   (TabbyAPI, text-generation-webui), 5001 (KoboldCpp) and 1337 (Jan). Each gets at
   most 3 seconds; no other host, path or request is tried, and nothing found is
   recorded. One server with a model list is taken, several are chosen by number,
   and when none answers you are asked for the base URL. A server answering 401
   or 403 is named, never used: give it with `--endpoint` and set
   `WERKBANK_API_KEY`. The search never sends an [API key](#api-key), and a
   server it found gets none afterwards either; a line `API key: sent ...` or
   `API key: not sent ...` says which applies when the variable is set. Under
   WSL2 an empty search adds that a server
   running under Windows needs `--endpoint` ([WSL2](PLATFORMS.md#wsl2)).
3. **Model.** Without `--model`, a single listed model is taken and several are
   chosen by number. A model listed by a file path or a model file's name (one
   with a backslash or ending in `.gguf`, `.safetensors`, `.bin`, `.pt` or `.pth`)
   is never offered: serve it under a name (llama.cpp `--alias`, vLLM
   `--served-model-name`). For an endpoint you gave, start reads its model list
   once, within 10 seconds.
4. **Server check.** The checks of [`werkbank preflight`](#check-the-server-first),
   with a cap of 32,768 unless `--max-tokens-per-call` is given. No chat request
   reaches the model before your choice, exactly one server and model found, or
   the flags. A server that is not ready stops start.
5. **Mode.** Start shows each mode with its cap and context, then
   asks for the mode and, for Quick, Normal and Full, `Three passes or one?`. `--mode` and
   `--passes` answer in advance; `--yes` without `--passes` runs three. It shows
   no time estimate: how long a pass takes follows from your server and model, and
   after the first pass start shows your own pace and the time left. The times of
   one measured setup are in the
   [measured effort](../METHODOLOGY.md#measured-effort).
   - *Cap:* `--max-tokens-per-call` if given. Otherwise 65,536, the largest cap,
     where the server reports a context
     that covers the mode's largest [context need](#selection-cap-and-context)
     with that cap, else 32,768. With 65,536 Quick needs up to 94,768 tokens,
     Normal up to 156,768 and Full up to 191,768, so a server with 128K of context
     gets that cap for Quick and 32,768 for Normal and Full, and one with 256K gets
     it for every mode. Full needs up to 159,000 tokens with 32,768, more than 128K
     holds.
   - *Context:* what the mode needs with the cap and whether the server's reported
     context covers it. The needs are measured ones
     ([context need](#selection-cap-and-context)); a model that carries more
     reasoning per step can need more.
6. **Passes.** After your confirmation (for example `Start Quick now: three
   passes with a cap of 65,536? [Y/n]`), start creates the series folder
   `runs/start-<YYYYMMDD-HHMMSS>-<quick|normal|full|trial>` under the current
   directory (a single pass: `-quick-single`, `-normal-single` or `-full-single`),
   or the new `--output DIR`,
   with mode 0700. Its name holds no model, host or endpoint, and start writes no
   file holding the endpoint address. Each pass is a `werkbank run` child process
   into `pass-1`, `pass-2` and `pass-3` (a single pass and the trial: `pass-1`
   only), with its progress on stderr; from the second pass on start shows its
   own measured pace.
   - A pass with an `invalid` result, an interrupted pass or one without a valid
     `run.json` failed technically. Its folder stays as evidence, and start asks
     `Repeat pass k now? [y/N]` once per pass. The repeat goes to
     `pass-k-repeat-1`; only the repeat is evaluated, and the result page and the
     summary name it with its reason.
   - A deadline stop and every other [censored](READING_RESULTS.md#other-terms)
     result count as measured and are never repeated.
   - When a technically failed pass is not repeated (`--yes`, your no, or a second
     failure; exit `2`) and after Ctrl-C during a pass (exit `1`), start prints
     the commands that finish the series by hand: the remaining passes into new
     folders, `werkbank compare` and `werkbank result` with a `--repeated` record
     for every repeated pass.
   - Passes whose recorded setups differ, so that they would not pool, stop the
     series with what changed (exit `2`).
7. **Evaluation.** `werkbank compare` of the passes into `evaluation/` and the
   result page into `result/`, both in the series folder. The summary names the
   tasks solved in each pass and the
   [subgoal fulfilment](READING_RESULTS.md#subgoal-fulfilment) overall and per
   category. A trial's summary counts its attempts by outcome instead.
8. **Submission.** After three passes, at a terminal and without `--yes`: `Send
   these results to the Werkbank maintainer? You will see exactly what is sent and
   confirm again. [y/N]`. `y` runs [`werkbank submit`](#submit-results) on the
   same terminal. Otherwise start prints the `werkbank submit` command for later.
   A trial and a single pass get neither the offer nor the command.

### Flags

| Flag | Meaning |
|---|---|
| `--endpoint URL` | The server's `/v1` base URL; skips the search. |
| `--model NAME` | The served model name to run. |
| `--mode quick\|normal\|full\|trial` | The mode. |
| `--passes 1\|3` | The number of passes of Quick, Normal or Full; refused with `--mode trial`. |
| `--yes` | Never ask. Needs `--mode`. Anything ambiguous, such as several servers or models, is refused with the flag to pass; a technically failed pass is not repeated; nothing is submitted. |
| `--browser` | Run the same steps in the [browser interface](#browser-interface): the command prints a local address to open, the other flags become the form's defaults, and every question is answered there. Not with `--yes`. |
| `--output DIR` | A new series folder. |
| `--max-tokens-per-call N` | The per-response cap, 1 to 65,536. |
| `--temperature`, `--top-p`, `--top-k`, `--stream`, `--serving-note FILE`, `--hardware-label TEXT`, and `--chat-template-kwargs JSON` or `--thinking LEVEL` | Passed unchanged to the server check and every pass, [as `run` takes them](#run-a-model). No thinking setting is sent unless you give one. |
| `--image IMAGE` | The container image of the readiness check and every pass. |

Arguments are refused as `run` refuses them, before any contact or output.
`--thinking` without `--model` is resolved once the model is chosen, before the
server check sends a request. Exit codes: [start](#exit-codes).

## Run a model

`werkbank run` runs the chosen cases as [Harbor](../INSTALLATION.md#harbor-runtime-python-312-and-docker)
trials, one per episode, into a new run directory and renders its report. Code
that a model submits runs only in the trial's containers, never on the host.
The whole plan is checked before any output exists or the endpoint is contacted.

```sh
.venv/bin/werkbank harbor-doctor
.venv/bin/werkbank preflight --endpoint http://127.0.0.1:8080/v1 --model MODEL_ID --max-tokens-per-call 65536
.venv/bin/werkbank run --all --endpoint http://127.0.0.1:8080/v1 --model MODEL_ID --max-tokens-per-call 65536 --output runs/local-001
.venv/bin/werkbank validate runs/local-001
```

| Flag | Meaning |
|---|---|
| `--endpoint URL` | Required. The server's HTTP(S) base URL, ending in `/v1`; a URL that ends in `/chat/completions` is refused. |
| `--model NAME` | Required. The exact model ID from the endpoint's `/v1/models`; `org/model` is fine. A value that looks like a local path is refused: serve the model under a name. |
| `--max-tokens-per-call N` | Required. The cap of one response, 1 to 65,536; at least 32,768 is recommended. |
| `--output DIR` | Required. A new directory. |
| `--case ID` (repeatable), `--set NAME` or `--all` | Exactly one is required ([selection](#selection-cap-and-context)). |
| `--temperature`, `--top-p`, `--top-k`, `--seed` | Sampling ([sampling](#sampling-and-the-recorded-setup)). |
| `--chat-template-kwargs JSON` or `--thinking LEVEL` | [Thinking and chat template](#thinking-and-chat-template). |
| `--stream` | Request server-sent events and record the time to the first token. |
| `--serving-note FILE`, `--hardware-label TEXT` | What the server does not report ([recorded setup](#sampling-and-the-recorded-setup)). |
| `--image IMAGE` | Another container image than the pinned one, by digest or image ID. |
| `--no-report` | Skip the report. |

Typical base URLs, to check against your server's configuration: llama.cpp
`http://127.0.0.1:8080/v1`, vLLM `http://127.0.0.1:8000/v1`, Ollama
`http://127.0.0.1:11434/v1`, LM Studio `http://127.0.0.1:1234/v1`. This is no
claim that each server is tested.

### Endpoint requirements

The endpoint must:

- return an integer `usage.completion_tokens` in every response, counting
  reasoning as you want it counted. Without it the case is `unsupported`; tokens
  are never estimated;
- return native tool calls with `finish_reason` `tool_calls`. A tool call labelled
  `stop` is `invalid`; nothing is repaired;
- accept requests without credentials, or with the one key you set in
  `WERKBANK_API_KEY`, sent as `Authorization: Bearer` ([API key](#api-key));
- with `--stream`: answer with `Content-Type: text/event-stream`, end the stream
  with `data: [DONE]` and honour `stream_options.include_usage`;
- with `--chat-template-kwargs` or `--thinking`: accept the request field
  `chat_template_kwargs`. A server that refuses it ends the first case as
  `invalid/endpoint_contract_error` and stops the run.

### API key

For a server of your own that asks for an API key, such as vLLM or llama.cpp
started with `--api-key`, set the environment variable `WERKBANK_API_KEY` in the
shell that runs the command:

```sh
read -rs WERKBANK_API_KEY   # type or paste the key; it is not shown
export WERKBANK_API_KEY
.venv/bin/werkbank start --endpoint http://HOST:8000/v1
```

`read -rs` keeps the key out of the shell's history, where a key written on a
command line would stay.

- **One source.** The variable is the only way to give a key: there is no flag,
  no file and no field in the browser interface. Unset or empty means no key, and
  then no request changes.
- **Shape.** 1 to 4,096 printable ASCII characters without a space. Any other
  value is refused before anything is contacted or written, with `WERKBANK_API_KEY
  is not a usable key` (exit `2`); the message never shows the value.
- **Where it goes.** As `Authorization: Bearer <key>` with every request to the
  endpoint you named: the chat requests, the model list and the one read of the
  server's version or settings. `werkbank run` and `werkbank preflight` send it
  to their `--endpoint`. `werkbank start` and the browser interface send it only
  to a server you named, with `--endpoint` or as a base URL you typed after an
  empty search, and say so with the line `API key: sent with every request to
  this server (WERKBANK_API_KEY)`. The search of this computer never sends it,
  and a server that the search found gets none of it afterwards. A redirect is
  never followed, so the key reaches no other host.
- **Never written down.** No run, journal, report, export, evaluation, result
  page or submission holds the key or says that one was used, and no command that
  Werkbank prints holds it. `werkbank run` and `werkbank preflight` remove the
  variable from their own environment as they read it, so Docker and the
  containers never see it. When a server answers an error that repeats your key,
  as it stands or with its characters escaped, the text of that answer is not
  kept; the journal then holds an empty error text, the one trace that a key was
  in use.
- **A placeholder is no secret.** A key of fewer than eight characters, such as
  the `EMPTY` or `none` that a local server is often started with, is sent like
  any key, but Werkbank does not look for it in a server's answers or in a
  submission: so short a word stands in ordinary error texts.
- **Commands printed for a found server.** When `werkbank start` found the server
  by its search and a pass fails, the commands it prints to finish the series by
  hand start with `env -u WERKBANK_API_KEY`, so that they send no key either.
- **Unencrypted over `http://`.** To a server on another machine an `http://`
  address sends the key, like every request and response, unencrypted. Werkbank
  does not refuse that and does not ask: use `https://` or a tunnel where the
  network between the machines is not your own.
- **A refused key.** `werkbank preflight` names HTTP 401 and 403 with a hint: to
  set the variable when no key was sent, or that the server refused the key.
- **Submitting.** While the variable is set, `werkbank submit` also searches what
  it would send for the key's value and sends nothing that holds it
  ([privacy check](../SUBMISSION.md#privacy-check)).

The key is meant for a server you run yourself, also on rented hardware.
Werkbank is not built or tested for the endpoints of online providers.

### Selection, cap and context

- **Selection.** `--set NAME` selects a named case set, a fixed list: `quick-v3`
  (14 cases, the easy and medium ones that a 64K context can run), `normal-v1`
  (31 cases, every case that is not very hard) or `full-v1` (all 40 cases).
  `--all` selects every case, which is the list of `full-v1`; `--case` single
  cases. A security case (S02, S03, S04,
  S05 or S06) always runs both of its episodes, attack and benign. `werkbank
  list-cases` prints every case with its budgets, the planned bytes and the case
  sets as JSON.
- **Cap.** Choose a `--max-tokens-per-call` that fits the server's context after
  prompt and history. Nothing lowers it automatically.
- **Context need.** A case's context need is the largest input of one of its
  requests as measured with Qwen3.8-Flash-Next, plus 32,768 tokens reserved for
  one response, plus whatever a larger cap reserves beyond that. A model that
  carries more reasoning per step can need more. The recommended context is the
  largest need of the planned cases rounded up to 64K, 128K or 256K. With a cap
  of 32,768 that is 64K for `quick-v3`, 128K for `normal-v1` and 256K for
  `full-v1`; with a cap of 65,536 it is 128K, 256K and 256K.
- **Below the recommendation.** A cap below 32,768 or a smaller context is
  allowed. The report and an evaluation then name the cap, and the planned cases
  whose need exceeds the context the server reports.
- **Budgets.** Every episode has a budget of 262,144 generated tokens and a wall
  ceiling derived from it ([budgets](../METHODOLOGY.md#budgets-and-wall-time)).
  Runs usually end long before; in the worst case `--all` takes several days. A
  plan whose evidence could exceed 1 GiB is refused.
- **After a stop.** An `invalid` result or an agent or cleanup deadline stops the
  run, and the remaining cases are `not_run`. Stopping the client does not cancel
  generation on the server: make sure the server is idle before the next run.

### Sampling and the recorded setup

Without sampling flags no sampling setting is sent, and the server applies its
own default, recorded as "server default". `--temperature` (0 to 2), `--top-p`
(above 0, at most 1), `--top-k` (1 to 1,000,000) and `--seed` (0 to
4,294,967,295) are each sent and recorded only when set. Scores need three
repetitions unless `--temperature 0`.

After the plan is admitted, `run` reads what the server reports about itself:

- one `GET {endpoint}/models` and, depending on the engine the model list names,
  at most one further read-only request: vLLM `GET /version`, llama.cpp `GET
  /props` or SGLang `GET /get_server_info`. Kept are the model's list entry, the
  engine, its version, the context and, where reported, quantization, KV-cache
  type, speculative decoding and the default template variables. A reported
  local model path is stored as its last component and a SHA-256;
- for an endpoint on `127.0.0.0/8`, `::1` or `localhost` only, the product names
  and memory of this machine's NVIDIA GPUs via `nvidia-smi`.

A failed read is recorded with its reason, never retried, and changes no status.
Two flags state what the server does not report:

- `--hardware-label TEXT`: the serving hardware in your words, 1 to 64 ASCII
  letters, digits, spaces and `. _ + ( ) -`, starting with a letter or digit and
  not ending in a space. Use it where nothing is detected or the server runs on
  another machine, such as behind an SSH tunnel to a loopback port. The hardware a
  comparison shows is the label, otherwise the serving note's accelerator,
  otherwise the detected GPUs.
- `--serving-note FILE`: a JSON file of at most 16 KiB with `"format":
  "werkbank-serving-note-v2"` and any of the keys below. An unknown key or an
  invalid value exits `2` with the key and its rule, before any contact.

| Serving note key | Value |
|---|---|
| `serving_runtime`, `serving_version`, `quantization`, `kv_cache_dtype`, `speculative_decoding`, `chat_template`, `tool_call_parser`, `accelerator`, `driver_version`, `compute_runtime`, `serving_host_os`, `engine_commit` | printable text of 1 to 256 characters, not a local path |
| `model_revision` | 40 lowercase hex digits of a commit, or `sha256:` and 64 lowercase hex digits |
| `context_profile` | the context in tokens, an integer from 1 to 16,777,216 |
| `accelerator_memory_gib` | an integer from 1 to 65,536 |
| `memory_type` | `discrete` or `unified` |

Results are comparable within one Werkbank version and case set. Repetitions of
one recorded setup (what the server reported, the note, the label, the sampling
without the seed, the template variables, the cap, streaming, the container image
and the case set) pool automatically; differing setups never do. The setup is
what you and the server report, not authenticated
([comparability](../METHODOLOGY.md#comparability)).

### Thinking and chat template

Without these flags no thinking setting is sent: the server's own applies, and
every display says "server default". Where the server reports its default
template variables (SGLang), they are part of the recorded setup. Any other
server-side thinking setting belongs in the serving note's `chat_template` text.

`--chat-template-kwargs JSON` sends template variables with every chat request,
as the request's top-level `chat_template_kwargs`, exactly as given: nothing is
translated for an engine, added, or resent without them.

- The value is a JSON object of 1 to 8 variables.
- A name has letters, digits and `_`, at most 64, not starting with a digit, and
  is no engine argument (`messages`, `conversation`, `tools`, `documents`,
  `chat_template`, `add_generation_prompt`, `continue_final_message`, `tokenize`,
  `padding`, `truncation`, `max_length`, `return_tensors`, `return_dict`,
  `return_assistant_tokens_mask`, `tokenizer_kwargs`).
- A value is `true`, `false`, an integer from -2,147,483,647 to 2,147,483,647 or
  one word of 1 to 64 letters, digits, `_` and `-` that does not read `true`,
  `false` or `null`. No `null`, fraction, list or object.
- The argument has at most 4,096 characters and the variables at most 1,024 bytes
  as canonical JSON; duplicate keys, `NaN` and `Infinity` are refused.
- An invalid value exits `2` with the rule, never the value, before any output or
  contact.

The variables are part of the recorded setup: runs with different variables never
pool, and a run that sends them is another setup than one that does not. An
engine may replace its own defaults with the request's object rather than merge
them, so give every variable your setup depends on. Werkbank does not check that
the server's chat template reads them: a template ignores a variable it does not
know, llama.cpp reads them only with `--jinja`, and some servers drop unknown
request fields.

`--thinking LEVEL` is a shorthand for the variables of a recipe the package
ships. It is resolved from the served model name before anything is contacted,
and only the resolved variables are sent and recorded, exactly as if given with
`--chat-template-kwargs`. Giving both flags exits `2`.

| Family (served model names, any case) | `off` | `low`, `medium`, `high`, `xhigh` | Source |
| --- | --- | --- | --- |
| Qwen3.8 (`qwen/qwen3.8-*`, `qwen3.8-*`) | `{"enable_thinking": false}` | `{"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": LEVEL}` | chat template of `Qwen/Qwen3.8-27B-FP8` |

| Family (served model names, any case) | `off` | `medium` | `high` | Source |
| --- | --- | --- | --- | --- |
| Nemotron 3.5 (`nvidia/nvidia-nemotron-3.5-*`, `nvidia-nemotron-3.5-*`) | `{"enable_thinking": false}` | `{"enable_thinking": true, "medium_effort": true}` | `{"enable_thinking": true, "medium_effort": false}` | chat template of `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |

`--thinking` takes `off`, `low`, `medium`, `high`, `xhigh` or `max`, but only the
levels a family's recipe lists; no recipe lists `max`. A served model name no
recipe matches, or a level its recipe lacks, exits `2` naming the recipes: set the
variables with `--chat-template-kwargs` then.

The report, the evaluation and the result page show what was sent on a Thinking
line and count the responses that returned reasoning apart from their answer
("reasoning seen in N of M responses"). The count is information only: it changes
no status, pool or score, shows on or off, never the level, and a server that
returns its thinking inside the answer text counts as not seen.

### Where to look

- **stdout:** one JSON line with the path of `run.json`, every case's status and
  `core_score: null`. A run gives no score; an [evaluation](#evaluation-across-runs)
  does.
- **stderr:** progress lines, then one line per case with status, reason, output
  tokens, time and its outcome group (`[solved]`, `[model failure]`, `[budget
  reached]`, `[technical problem]`, `[technical problem (invalid)]` or `[not
  run]`), and a summary such as `outcomes: 9 solved, 2 model failure, 1 budget
  reached, 1 technical problem (1 invalid), 1 not run`.
- **The run directory**, created with mode 0700:

| Path | What it holds |
|---|---|
| `plan.json` | the run's identity: model, recorded setup, harness digest, the planned episodes |
| `tasks/` | one task package per episode, as the run used it |
| `episodes/` | one folder per episode with its evidence and verdict |
| `trials/` | Harbor's own files of each trial; they hold absolute host paths |
| `run.json` | the case results, written last; a run without it was interrupted |
| `report.md`, `report.html` | the [run report](READING_RESULTS.md#the-run-report), unless `--no-report` |

Every status, reason and outcome group is explained in the
[glossary](READING_RESULTS.md#glossary). Journals and code snapshots hold raw
model output and stay local unless you send them as raw data with a separate yes
in `werkbank submit`. Inspect a run before you publish it.

### Check the server first

`werkbank preflight` checks the [endpoint requirements](#endpoint-requirements)
before a long run. It takes the endpoint arguments of `run` (`--endpoint`,
`--model`, `--max-tokens-per-call`, the sampling flags,
`--chat-template-kwargs` or `--thinking`, `--stream`, `--serving-note` and
`--hardware-label`), so a run's command line works without `--output`, `--case`,
`--set`, `--all`, `--image` and `--no-report`.

It contacts the model and writes no file. It refuses arguments as `run` does,
before any contact. It reads what a run reads about the server
([recorded setup](#sampling-and-the-recorded-setup)), then sends at most two chat
requests the way an episode sends them, with your sampling flags, template
variables, streaming mode and cap, each under a 300-second deadline and never
retried:

1. **Tool call (request 1):** one prompt and one tool, `record_check`, asking for
   one call. Passed with a native call of that tool, `finish_reason` `tool_calls`
   and arguments that are a JSON object.
2. **Tool result (request 2):** sent only after request 1 and its token count
   passed, with the call and its result in the history as an episode sends them.
   Passed when the server accepts it.
3. **Token count:** every reply carries an integer `usage.completion_tokens`.

stdout holds a plain-text summary:

- the server and its version where reported, and whether `/v1/models` lists the
  model;
- the reported context and, for each case set, whether it
  covers it with the checked cap or which cases need more, with the context
  recommended for them. A covered selection reads `covers the measured need for
  ...`: the [context needs](#selection-cap-and-context) were measured with
  Qwen3.8-Flash-Next (EXL3 3.05 bpw) on Velo 0.7.7;
- each check as passed, failed or not checked, with its reason and the status a
  run records for the same outcome;
- a Thinking line with the template variables sent ("server default" without
  any) and, once a reply arrived, "reasoning seen in N of M replies (information
  only; the level is not checked)";
- `Ready: yes` or why not, and hints from a fixed table for vLLM, llama.cpp,
  SGLang, Ollama and LM Studio: the identified server's line, or every server's
  where none is identified. The hints are advice, not tested against every server
  version.

The output never shows a response body, model output, the endpoint URL or a
credential: with an [API key](#api-key) set, the fingerprint and both requests
carry it, and on HTTP 401 or 403 the hint says whether a key was sent, never the
key. A passed preflight is no guarantee for a run, whose cases send longer
prompts and other tools. A request that ran into its deadline may keep generating
on the server: wait until the server is idle before a run.

## Validate, regrade and export

```sh
.venv/bin/werkbank validate runs/run-1
.venv/bin/werkbank regrade --source runs/run-1 --output runs/run-1-regrade
.venv/bin/werkbank validate runs/run-1-regrade --source runs/run-1
.venv/bin/werkbank export --source runs/run-1 --output runs/run-1-export
```

A separate verifier container grades each episode by replaying its journal.
Whatever it cannot confirm is `invalid/evidence_ungradable` and stops the run.

| Command | What it does | Writes |
|---|---|---|
| `werkbank validate RUN` | Checks a run's records and evidence against each other, offline and without Harbor. It repairs nothing. | nothing; one JSON line on stdout |
| `werkbank validate REGRADE --source RUN` | Checks a regrade against the run it regraded. | nothing |
| `werkbank validate EVAL [--source RUN ...]` | Checks an [evaluation](#evaluation-across-runs). | nothing |
| `werkbank regrade --source RUN --output NEW` | Grades the recorded evidence again in fresh containers, without an endpoint; code submissions are executed again. The source stays unchanged. | `regrade.json` in the new directory |
| `werkbank export --source RUN --output NEW` | Packs the plan, the run record, the task packages and every episode's evidence for a regrade elsewhere. The same run always gives the same bytes. `trials/` and the report are left out. | `run.tar` and `export.json` |

- **Validation checks completeness and consistency, not authenticity.** It is no
  defence against someone who rewrites a run directory coherently. Only a
  regrade grades the recorded evidence again.
- **A regrade needs the build that wrote the run.** `regrade` refuses a run whose
  `plan.json` names another `harness_sha256` (exit `2`), naming both digests.
  `werkbank doctor --offline` prints this build's.
- **An export** is extracted into a new directory and regraded there.
- **Another build's run** is read by `validate`, `report`, `export` and `compare`
  only while its budgets, limits and cases equal this build's.

## Reports and evaluation

`werkbank run` renders `report.md` and `report.html` into the run directory;
`werkbank report RUN` renders them for a run that skipped it. Both validate the
run first and never overwrite a report. A report shows one run: its setup, every
case's status, reason, outcome group, tokens, requests and time, the subgoals
reached and the effort per category. It gives no overall or category score.
[The run report](READING_RESULTS.md#the-run-report) explains each part.

### Evaluation across runs

`werkbank compare RUN [RUN ...] --output NEW` evaluates complete runs offline,
without Harbor or an endpoint. It validates every run, refuses an interrupted run
and a run given twice, and writes a new directory, never into a run. Every
quantity and rule is defined in the [methodology](../METHODOLOGY.md).

```sh
.venv/bin/werkbank compare runs/run-1 runs/run-2 runs/run-3 --output runs/eval-1
.venv/bin/werkbank validate runs/eval-1 --source runs/run-1 --source runs/run-2 \
  --source runs/run-3
```

| Flag | Meaning |
|---|---|
| `--output NEW` | Required. The new evaluation directory. |
| `--set NAME` | Evaluate every run on exactly that case set's cases, so larger runs pool and rank with runs of the set: `--set normal-v1` or `--set quick-v3` on Full runs gives their Normal or Quick values. A run that does not plan every case of the set is refused. |
| `--weight-profile FILE` (repeatable) | Rescore the overall points with your own category weights ([weight profiles](../METHODOLOGY.md#weight-profiles)); format `werkbank-weight-profile-v2`, `schemas/weight-profile.schema.json`. |
| `--resamples N` | Bootstrap resamples, 100 to 100,000; default 10,000. |
| `--bootstrap-seed N` | Bootstrap seed, 0 to 4,294,967,295; default 0. |

| File | What it holds |
|---|---|
| `evaluation.json` | the record (`werkbank-evaluation-v9`, `schemas/evaluation.schema.json`), written last; it binds the other files by digest |
| `evaluation.md`, `evaluation.html` | every table: pools, scores and bounds, outcomes, failure classes, effort, comparisons |
| `model-card.md` | a model card of the evaluated pools |
| `dataset/` | the tables as CSV with a dataset card, in a Hugging Face layout |

The CSV files are RFC 4180 (CRLF, UTF-8 without BOM, an empty cell for no value)
and hold the data verbatim: they are not neutralised for spreadsheet formulas.

stdout holds one JSON line with the path of `evaluation.json` and the number of
pools. stderr shows per pool its model, the tasks solved on average per pass with
the range of the passes and the
[subgoal fulfilment](READING_RESULTS.md#subgoal-fulfilment) overall and per
category, or why the pool is not evaluated.

- **Pools.** Repetitions of one [recorded setup](#sampling-and-the-recorded-setup)
  pool automatically. Model, server, hardware, sampling, thinking, context, cap
  and streaming are shown, never a reason to exclude a result. Without
  temperature 0 a pool needs three runs for scores.
- **Scores.** A pool on exactly the cases of `normal-v1` or of `full-v1` that
  measured every one of them has a weighted overall score over that set, and the
  evaluation names the set beside it; scores on different sets do not compare.
  Any other case list, `quick-v3` among them, gives score bounds against all 40
  cases and no point score. Censored attempts count as not solved.
- **Validation.** `werkbank validate EVAL` checks format, files, digests and
  internal consistency. With `--source RUN` for exactly the evaluated runs, the
  build that wrote the evaluation also recomputes it.
- **Another format.** An evaluation written in another format is refused, also by
  `werkbank result` and `werkbank submit`: write it again from its runs with
  `werkbank compare`.

### Result page

`werkbank result EVAL --output NEW` writes an evaluation's result page offline
into a new directory beside the evaluation, never inside it. It renders only an
evaluation this build wrote, and the same input always gives the same bytes.

```sh
.venv/bin/werkbank result runs/eval-1 --output runs/result-1
```

| File | What it holds |
|---|---|
| `index.html` | the page: one static, self-contained file without scripts |
| `result.json` | the record (`werkbank-result-v1`): binds the page to `evaluation.json` by digest, records the repeated passes and marks a trial or a single pass |

| Flag | Meaning |
|---|---|
| `--repeated PASS:SEED:KIND[:STATUS/REASON]` (repeatable) | Discloses a pass that was repeated. `KIND` is `technical_result` with `invalid/REASON`, `interrupted` or `no_run_record`. `werkbank start` prints the flag for a series finished by hand. A pass or seed outside the evaluation's one pool is refused. |
| `--trial` | Marks the page as a trial: no benchmark and no score, with `"trial": true` in `result.json`. Only an evaluation of one pass in one pool can be one. |
| `--single-pass` | Adds the notice that the page shows one pass instead of three, with `"single_pass": true` in `result.json`; the numbers stay. The same rule on the evaluation applies. Not with `--trial`. |

`werkbank start` passes `--trial` and `--single-pass` itself, and `werkbank
submit` refuses a page with either mark. What the page shows:
[reading your results](READING_RESULTS.md#the-result-page). Where its numbers
stop: [results and their limits](LIMITS.md).

## Submit results

`werkbank submit EVAL --result DIR [--source RUN ...] [--write FILE]` sends an
evaluation's results once to the Werkbank maintainer, only after you saw exactly
what is sent and typed `y` at a terminal. Piped or redirected input never
confirms.

```sh
.venv/bin/werkbank submit runs/eval-1 --result runs/result-1 \
  --source runs/run-1 --source runs/run-2 --source runs/run-3
```

| Flag | Meaning |
|---|---|
| `--result DIR` | Required. The evaluation's result page directory from `werkbank result`. |
| `--source RUN` (repeatable) | An evaluated run, in any order. Needed for a complete check and to offer raw data. |
| `--write FILE` | Write the exact archive to a new local file instead of sending it. |

1. It asks five optional multiple-choice questions.
2. It checks the evaluation, the result page and, with `--source`, the runs, and
   runs the privacy check.
3. It shows a summary: per pool the served model name, serving settings, hardware
   and image digest; the case set, the runs, the files with their sizes, the
   installation ID, the receiver's address and the consent text.
4. Only your `y` sends, in one HTTPS request without retry, proxy or redirect.

Always sent are the result page, the evaluation directory and an envelope with
your answers and a random installation ID stored on this computer. The runs'
export archives, which hold every model response, are sent only with a separate
yes. Never sent are the endpoint address, host or user names, paths or
credentials. A trial and a single pass are refused.

This build sends to `https://submit.victorswerkbank.workers.dev/submissions` and
names `werkbank-loeschung.jeep654@passinbox.com` for deletion requests. Format,
privacy check, receiver answers, retention and deletion:
[submitting results](../SUBMISSION.md).

## Browser interface

`werkbank-ui [DIRECTORY ...] [--port N] [--read-only]` serves Werkbank in the
browser on `127.0.0.1` and prints the address with this session's token. It runs
the [guided start](#guided-start) with its questions as forms and shows the run
and evaluation directories under its arguments.

```sh
./werkbank --browser
.venv/bin/werkbank-ui
.venv/bin/werkbank-ui --read-only runs/
```

| Argument | Meaning |
|---|---|
| `DIRECTORY ...` | Run or evaluation directories, or directories holding them; default `runs`. New series go under the first one. |
| `--port N` | The loopback port; default a free one. |
| `--read-only` | The view alone: nothing is started and nothing is written. Needs a `DIRECTORY`. |

`werkbank start --browser` serves the same with the start's flags as the form's
defaults. A series runs in the serving process: Ctrl-C there first stops the
running pass. Details and security controls: [browser interface](INTERFACE.md).

## Exit codes

| Command | `0` | `1` | `2` | `3` |
|---|---|---|---|---|
| `run`, `regrade` | every selected case measured (`passed`, `failed`, `output_limit`, `timeout`) and every security case completely measured (both episodes `passed` or `failed`) | | a case is `invalid`, the selection or output cannot start, or `run` cannot render its report | no case `invalid`, but one `unsupported` or `not_run`, or a security case incompletely measured |
| `preflight` | every check passed | | a run would stop (a request ended in what a run records as `invalid`: the server's answer, a failed or broken connection, or a failed request worker), or the arguments are refused | cases would be lost or readiness is not confirmed: a missing token count, a capacity rejection, a reply without a native tool call or with another tool or non-object arguments, `finish_reason` `length` or `content_filter`, or no reply within the deadline |
| `validate RUN` | valid and complete | | invalid | valid but interrupted (no `run.json`) |
| `validate REGRADE` | valid | | invalid, or not exactly one `--source` | |
| `validate EVAL` | completely checked | | invalid | valid, but checked without its sources or by another build |
| `compare` | the evaluation is written | | an input is invalid or interrupted, a run is given twice, a run does not plan every case of the `--set`, a weight profile is invalid or given twice, or the output is not new | |
| `report`, `export`, `result`, `harbor-smoke` | written | | refused or failed | |
| `harbor-doctor` | ready | | not ready | |
| `doctor --offline`, `list-cases` | always | | | |
| `start` | every pass measured completely and the result page written | you stopped it: a declined prompt, end of input, or Ctrl-C at a prompt or during a pass | refused or failed: arguments, readiness, no server or an ambiguous server or model under `--yes`, a server check that is not ready, a technically failed pass that is not repeated, passes of differing setups, or a failed evaluation or result page | the series is evaluated, but a pass left cases unmeasured |
| `submit` | the receiver confirmed its receipt, or the file was written or already holds the archive | | refused before the confirmation; something needs fixing | nothing was confirmed: declined, end of input, Ctrl-C, or no confirmed receipt |
| `werkbank-ui` | served until Ctrl-C | | it cannot serve, or an argument is refused | |

`install.sh`: `0` ready; `1` a prerequisite is missing, `.venv` is refused or you
declined to have Python provided, and nothing was changed; `2` a step failed, also
a download of uv that does not match its pinned hash: fix the cause it shows and
run it again. With `--check`: `0` installed completely from this checkout's lock
files, `1` otherwise.

`./werkbank`: the code of `install.sh` when the installation stops (`1` or `2`),
else the code of `werkbank start`.

- Only `0` means a complete measurement. Exit codes never express comparability
  or difficulty.
- `preflight`: the reported context and the model list never change the code.
- `compare`: what a pool cannot evaluate is marked in the evaluation, never in
  the exit code. When the evaluation would fail its own checks, no output
  directory is created.
- `validate`: `--source` belongs to a regrade (exactly once) or an evaluation
  (repeatable); `validate RUN --source ...` is refused with `2`.
- `start`: the final question whether to submit never changes the code. A no, end
  of input or Ctrl-C there counts as no.
