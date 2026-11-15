# Licensing

Victor's Werkbank is open source. Its code is licensed under the
[Apache License, Version 2.0](LICENSE), its documentation under the
[Creative Commons Attribution 4.0 International License](LICENSES/CC-BY-4.0.txt)
(CC BY 4.0), and its tasks are available under either of the two, at your choice.
The package metadata gives the SPDX expression
`Apache-2.0 AND CC-BY-4.0 AND (Apache-2.0 OR CC-BY-4.0)` for what a distribution of
the package holds: the code under Apache-2.0, the documentation under CC BY 4.0 (the
README that the package metadata shows as its description and the Markdown files
that a source distribution carries beside the package) and the tasks under
Apache-2.0 or CC BY 4.0. The table below says which licence covers which file.

## Which licence covers which file

The first row that matches a file applies, so a Markdown file inside a task package,
such as its `instruction.md`, belongs to the tasks.

| Files | Licence | SPDX |
|---|---|---|
| The tasks: every file under `src/victors_werkbank/domains/` (the domain modules, which hold each task's prompt, tools, fixtures, simulator and outcome check) and under `src/victors_werkbank/tasks/` (the task packages, one per episode) | Apache-2.0 or CC BY 4.0, at your choice | `Apache-2.0 OR CC-BY-4.0` |
| Every other Markdown file (`*.md`), such as this page, the [README](README.md), the documentation under `docs/`, the [changelog](CHANGELOG.md) and the [security policy](SECURITY.md) | CC BY 4.0 | `CC-BY-4.0` |
| Every other file, such as the rest of the package under `src/`, the schemas, profiles and examples, `install.sh`, the launcher `werkbank`, the build and requirements files, `NOTICE` and `CITATION.cff` | Apache-2.0 | `Apache-2.0` |

[LICENSE](LICENSE) and [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt) are the
licences' own texts and keep their publishers' terms.

## Why the tasks have two licences

The task content lives in code. A domain module holds a task's prompt text, its
tools, its fixtures, the simulator that answers the model's tool calls and the check
of the outcome, all as Python; a task package holds the prompt as `instruction.md`
beside the verifier's entry point and the container files that run it
([datasheet](docs/DATASHEET.md#composition)). As software the tasks belong under a
software licence, and as a task set (prompts, fixtures and expected outcomes) they
are content, for which Creative Commons licences are the usual terms. Either licence
lets you use, change and pass on the tasks: Apache-2.0 suits reusing them as code,
for example in another harness, and CC BY 4.0 suits reusing them as data, for
example in a collection of evaluation sets. What you pass on follows the terms of
the licence you chose.

## Attribution

[NOTICE](NOTICE) names the copyright holder, The Victor's Werkbank Authors (Victor
Klaue).

- **Under Apache-2.0**, a redistribution keeps [LICENSE](LICENSE) and
  [NOTICE](NOTICE) and marks the files it changed (section 4 of the licence).
- **Under CC BY 4.0**, credit The Victor's Werkbank Authors, name CC BY 4.0 with a
  link to the licence, link to the repository you took the material from and say
  whether you changed it (section 3 of the licence).
- **To cite Werkbank** in a paper, a report or a model card, use
  [CITATION.cff](CITATION.cff); GitHub offers it as "Cite this repository". Its
  licence field names the code's licence, Apache-2.0; the full mapping of licences
  to files is the table above.

## The runtime image

The container image `ghcr.io/aisyndicate-ch/werkbank-runtime` holds third-party
software under its own licences: Python, the Rust toolchain and, from Debian, GCC and
the C library under the GPL and LGPL. None of it is part of Victor's Werkbank, and
the licences above do not cover it. [docker/runtime/Dockerfile](docker/runtime/Dockerfile)
names the pinned base images and the Debian snapshot, where the sources of every
package stay available.

## Contributions

Contributions come as issues for now ([README](README.md#issues)). Where a change
to a file is contributed, it is made under the licences of the files it changes,
inbound as outbound: code under Apache-2.0 (as section 5 of that licence states),
documentation under CC BY 4.0, and a change to a task under both, so the tasks stay
available under either. A new file falls under the row of the table above that
matches it. Contributions follow the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): every contributed
commit carries a `Signed-off-by:` line, with which its author certifies the right
to submit the change under the project's licences. There is no separate contributor
licence agreement.

## No third-party tasks

Every task, fixture and grader was written for this benchmark. No third-party
benchmark task, text, data, fixture or grader is bundled, and published research
informed some designs as inspiration only. Later tasks keep it that way: synthetic,
redistributable fixtures only and no imported benchmark tasks. What the
installation fetches (Harbor, the Python packages of the requirements files and the
pinned container image) is not part of this repository and keeps its own licences.

## Training and the canary

The tasks are here to measure models, not to train them. Please do not use the
tasks, their fixtures, their graders or the documents that describe them to train
or fine-tune a model, and keep them out of training corpora. The files that hold
the tasks and their answers carry a canary string, so that a corpus can filter them
out; please keep it in those files when you pass the tasks on. The
[datasheet](docs/DATASHEET.md#contamination) says where the canary is, where it is
not and what it shows. This is a request, not a condition of the licences: the
licences above are the only terms.

## Results and submissions

The licences cover the files of this repository, not the results you measure with
Werkbank. This version has no leaderboard, and a result describes one setup for
one Werkbank version and one case set
([what results show and what they do not](docs/users/LIMITS.md)).

Nothing reaches the maintainer automatically. `werkbank submit`, and the offer at
the end of `werkbank start`, send a submission only after you have seen exactly what
it holds and confirmed it, at a terminal or in the browser interface, never with
`--yes`, and a submission stays self-reported
([submitting results](docs/SUBMISSION.md)). No other part of Werkbank contacts the
maintainer.
