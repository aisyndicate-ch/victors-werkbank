# Security policy

Report a security problem privately, never in a public issue or discussion.

## Supported versions

The 0.3.x versions get security fixes, each in a new 0.3.x release. A fix in the
package `victors_werkbank` changes the harness digest, so results of the fixed
release are compared only with each other
([comparability](docs/METHODOLOGY.md#comparability)).

## Reporting a vulnerability

Use either private channel:

- **GitHub private vulnerability reporting**: the repository's Security tab,
  "Report a vulnerability"
  ([direct link](https://github.com/aisyndicate-ch/victors-werkbank/security/advisories/new)).
  The report and the conversation about it are visible only to you and the
  maintainer.
- **E-mail**: victorswerkbanksec.dentist951@passinbox.com.

Do not send credentials, real endpoint addresses or private model inputs: replace
an address with `ENDPOINT` and remove any credential. Where you can, reproduce the
problem against a fake endpoint, which needs no real model.

## What to include

- The output of `werkbank doctor --offline`: the version, the harness digest and
  your platform. The command contacts nothing.
- The affected part: a command, a module, a workflow or a release file.
- The steps that reproduce the problem, with the command, its exit code and the
  files involved.
- What an attacker controls (model output, a task or tool text, a run directory, a
  network position) and what they gain: which boundary of the [scope](#scope) is
  crossed.
- Whether and when you plan to disclose it, and whether and how you want to be
  credited.

## After a report

1. The maintainer confirms that the report arrived and assesses it. Keep the
   details private meanwhile.
2. You are told whether it is accepted as a vulnerability and, if not, why. A
   problem outside the [scope](#scope) can become a public issue with your
   agreement.
3. An accepted vulnerability is fixed in a new 0.3.x release. The maintainer
   agrees the disclosure date with you and publishes a GitHub security advisory
   with the release, crediting you if you wish; where it applies, the advisory
   requests a CVE.

## Scope

In scope are this repository's code, its workflows and its release files. These
are vulnerabilities:

- **Code execution.** Code that a model writes runs only in containers, never on
  the host and with no host fallback: as a fresh unprivileged user under resource
  limits, without network beyond loopback, and a separate container verifies the
  result. A way to run model-written code on the host, to leave the container or
  to reach the hidden tests or expected outcomes from it, at compile time or at
  run time.
- **Grading.** A forged pass that a regrade accepts, or model output, tool text or
  a task file that changes the runner's clocks, journal or results.
- **Privacy.** An endpoint address, a credential, a host or user name or a local
  path that reaches a run record, report, export, result page or submission; a
  submission that is sent without your explicit confirmation or to another
  address; identifying data that the
  [privacy check](docs/SUBMISSION.md#privacy-check) should have refused.
- **Network.** A request that uses a proxy, follows a redirect or is retried, or
  any contact beyond those the command reference lists
  ([what the commands contact](docs/users/CLI.md#what-the-commands-contact)).
- **Browser interface.** A request from another website, another host or another
  user of this computer that reads the interface or acts through it
  ([security](docs/users/INTERFACE.md#security)).
- **Supply chain.** A workflow that exposes a token or runs untrusted code with
  write permissions; a [release file](#release-files) whose checksum, software
  bill of materials or build provenance attestation does not match what the
  release workflow built; a [runtime image](#runtime-image) at the pinned digest
  that holds what its Dockerfile does not install. The workflows run on
  GitHub-hosted runners with read-only default permissions, hold no model
  credential and never use `pull_request_target`.

These are not:

- A coherent rewrite of a run directory by someone with write access to it.
  `werkbank validate` checks a run directory for completeness and consistency, not
  for authenticity; only a regrade (`werkbank regrade`) grades the recorded
  evidence again.
- The limits of the container boundary: the containers share the host's kernel,
  and nothing is claimed for a hostile multi-tenant host.
- A model that follows an injected instruction, leaks data or crosses a boundary
  in a security task. That is the result the task measures.
- A vulnerability in an inference server, Harbor, Docker, a Python package or an
  upstream part of the runtime image, unless the way Werkbank uses it exposes it.
  Report it to its project.

## Runtime image

Code tasks run in one container image that the maintainer builds and publishes:
`ghcr.io/aisyndicate-ch/werkbank-runtime` in the GitHub Container Registry, public
and pulled without credentials. This version pins the image tagged `0.3.0-1`; the image has version numbers of
its own. A
run names it by the digest of its multi-architecture index, never by the tag, and
records that digest.

- **Where it comes from.** [`docker/runtime/Dockerfile`](docker/runtime/Dockerfile)
  is the whole recipe. It starts from the official Python image and copies only
  the compiler, its libraries and the standard library from the official Rust
  image, both pinned by digest. gcc and libc6-dev come from a dated snapshot of
  the Debian archive, so a rebuild installs the same versions, and apt verifies
  them against the archive keys. cargo, rustup and rust-lld are left out, and no
  toolchain file is writable for other users. The build compiles and runs a test
  program on every architecture.
- **What verifies it.** The pinned digest: Docker refuses any other content. The
  image carries no signature and no attestation of its own, so the digest in this
  repository, `SMOKE_IMAGE` in
  [`src/victors_werkbank/harbor_smoke.py`](src/victors_werkbank/harbor_smoke.py),
  is what counts. A version tag names one image and is never built again.
- **Vulnerabilities.** A vulnerability in a part of the image is in
  [scope](#scope) when submitted code can use it to cross a boundary named there;
  otherwise report it to the part's own project. The image is never updated in
  place: its packages stay at their pinned versions so that results stay
  comparable, and a known vulnerability in them can remain for as long as no
  submitted code can reach it. A fix is a new image, a new digest and a new 0.3.x
  release, and runs on different images are never pooled.

## Release files

A release holds the wheel, the sdist, an SPDX software bill of materials
(`sbom.spdx.json`) and their checksums (`SHA256SUMS`). The release workflow of
this repository builds them from the version tag, and for each of the four files
GitHub keeps a build provenance attestation that says so. In the directory with
the downloaded files:

    sha256sum --check --ignore-missing SHA256SUMS
    gh attestation verify FILE --repo aisyndicate-ch/victors-werkbank

The first command checks the files against the checksums; the second fetches the
attestation of one file from GitHub and checks it. A file that fails either check
did not come from the release workflow; report it as above.

## No bounty

There is no bug bounty and no paid reward. The advisory credits reporters who wish
it.
