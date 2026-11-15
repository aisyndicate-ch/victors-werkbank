# Installation

Werkbank installs into a `.venv` inside its own folder, from lock files that pin
every package by hash, and pulls one container image pinned by its digest. The
installation needs the network and contacts no model.

## Requirements

| What | Needed |
|---|---|
| Operating system | Linux, or Linux inside WSL2 on a Windows host ([platforms](users/PLATFORMS.md)). macOS and native Windows are not supported. |
| Python | 3.12 or newer with its `venv` module. Without one, the installation offers to [provide Python](#python-on-demand) inside the folder. |
| Docker | Docker Engine whose `docker` command works for your user, for example as a member of the `docker` group. |
| Runtime image | Pulled by the installation: about 300 MB to download and 830 MiB unpacked ([Harbor runtime](#harbor-runtime-python-312-and-docker)). |
| git | Only to clone and update a checkout. A [release archive](#from-a-release-archive) needs none. |

A model server is needed only for a run, not for the installation.

## One step

In the root of a checkout:

```sh
./werkbank                # installs on its first call, then runs the guided start
```

`./werkbank` runs `./install.sh` on its first call, and again whenever
`./install.sh --check` finds no complete installation; it stops with the script's
exit code when the installation stops. Then, and on every later call directly, it
runs the [guided start](users/CLI.md#guided-start) with every argument passed on
unchanged. The installation alone, without the start:

```sh
./install.sh
.venv/bin/werkbank start  # the guided start; it contacts your model server
```

`install.sh` does this, in order:

1. It checks Python, its `venv` module and your access to Docker, and names every
   missing one with its fix. Nothing has been changed at that point.
2. It creates `.venv` in the checkout.
3. It runs the four installation steps shown [by hand](#by-hand): three from lock
   files that pin every package by hash, then the package itself.
4. It runs `werkbank doctor --offline`.
5. It pulls the pinned container image by its digest.
6. It runs `werkbank harbor-doctor`.

It never asks for administrator rights. It writes files only inside the checkout,
apart from pip's download cache and Docker's image store. It downloads nothing
that a lock file, the image digest or a hash in the script does not pin. It asks
one question, and only when no usable Python is found
([Python on demand](#python-on-demand)).

| Option | Effect |
|---|---|
| `--skip-docker` | Installs the Python environment only: no Docker check, no image pull and no `harbor-doctor`. Cases cannot run until the script runs again without it. |
| `--check` | Changes and prints nothing. Exit `0` when a complete run without `--skip-docker` installed `.venv` from this checkout's lock files, pulled the image and passed both checks, else `1`. `./werkbank` asks this. |
| `-h`, `--help` | Shows the options. |

- **Interpreter:** `WERKBANK_PYTHON` if set, else the first of `python3`,
  `python3.13` and `python3.12` that is 3.12 or newer with its `venv` module, else
  the one an earlier run [provided](#python-on-demand).
- **Running it again is safe.** A `.venv` it installed from the same lock files is
  reused, and only the package's own installation, the image pull and the checks
  run again. After an interruption or a failed step it continues where it stopped.
- **A `.venv` it did not create** is refused: keep it and install [by hand](#by-hand),
  or remove it and run the script again. One it installed from other lock files
  is refused as well: remove it and run the script again.
- **Exit codes:** [`install.sh` and `./werkbank`](users/CLI.md#exit-codes).

### Python on demand

When no Python 3.12 or newer with its `venv` module is found, `WERKBANK_PYTHON` is
not set and the missing Python is the only problem left, `install.sh` asks at a
terminal whether to provide one inside the checkout: `Provide Python 3.12.15 this
way? [y/N]`. Enter, or anything but `y` or `yes`, stops it with exit `1`, and
nothing is changed. Without a terminal it does not ask and stops as for any
missing prerequisite. Before it asks, it says exactly what it would load:

1. uv 0.13.5, the release archive for this machine
   (`uv-x86_64-unknown-linux-gnu.tar.gz` or `uv-aarch64-unknown-linux-gnu.tar.gz`)
   from `https://github.com/astral-sh/uv/releases/download/0.13.5/`, with the
   `urllib` of any `python3` on the machine, else with `curl`, over HTTPS only. Its
   SHA-256 is pinned in `install.sh` and checked before the archive is unpacked; a
   mismatch removes it unused and stops with exit `2`.
2. CPython 3.12.15, which that uv installs from the releases of
   [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
   and checks against the SHA-256 values built into it. uv runs without its
   settings files, Python mirrors or a cache outside the checkout and installs no
   commands on `PATH`.

Both land in `.werkbank-tools/` in the checkout, and `.venv` is created with that
Python.

## From a release archive

Without git, install from a release instead of a checkout. The release archive is
the sdist, `victors_werkbank-<version>.tar.gz`, among the release files. Check it
against `SHA256SUMS` as [release files](../SECURITY.md#release-files) describes,
then unpack it and run the launcher in the unpacked folder:

```sh
tar -xzf victors_werkbank-<version>.tar.gz
cd victors_werkbank-<version>
./werkbank
```

The archive holds the launcher, `install.sh`, the lock files and the sources, so
the installation runs as from a checkout. A new version is a new archive in a new
folder, with a `.venv` of its own.

## By hand

The same steps typed, in an activated environment. Then pull the image and run
`harbor-doctor` as in the [next section](#harbor-runtime-python-312-and-docker).

```sh
python3 -m venv .venv
. .venv/bin/activate  # uv_build locates its pinned executable on PATH
python -m pip install --require-hashes --index-url https://pypi.org/simple -r requirements-dev.txt
python -m pip install --require-hashes -r requirements-harbor-build.txt
python -m pip install --no-build-isolation --require-hashes -r requirements-harbor.txt
python -m pip install --no-deps --no-build-isolation -e .
werkbank doctor --offline
```

`install.sh` refuses a `.venv` installed this way, as it did not create it;
`./werkbank` therefore does not start from it. Use `.venv/bin/werkbank start`.

## Harbor runtime: Python 3.12 and Docker

Cases run as [Harbor](https://github.com/harbor-framework/harbor) trials in Docker
containers. Runs need the pinned Harbor 0.24.0, which the steps above install,
Docker, and the pinned runtime image. The image is
`ghcr.io/aisyndicate-ch/werkbank-runtime` in the GitHub Container Registry, public
and pinned by the digest below. It holds Python 3.12 and the Rust compiler and is
built from [`docker/runtime/Dockerfile`](../docker/runtime/Dockerfile)
([where it comes from](../SECURITY.md#runtime-image)).

```sh
docker pull ghcr.io/aisyndicate-ch/werkbank-runtime@sha256:2c57f1598aa42b357d8220880591ca5c09939621ed5c12d12c05a1f6c0521f36
.venv/bin/werkbank harbor-doctor
.venv/bin/werkbank harbor-smoke --output /tmp/werkbank-harbor-smoke
```

`install.sh` runs the first two. The third is optional: `harbor-smoke` runs one
task (M02) as a Harbor trial against a built-in fake endpoint on this computer and
grades it again from its recorded evidence, into a new output directory. It
contacts no model and measures none. Harbor fetches its own helper images on
first use, so the first `harbor-smoke` or run still needs the network.

The package itself declares no dependencies. Without Harbor, `werkbank run` exits
`2` before any output exists or the endpoint is contacted, and the offline
commands still work.

## The two checks

`werkbank doctor --offline` prints the installed version, the harness digest
(`harness_sha256`), Python and the platform as JSON. It contacts nothing and
exits `0`.

`werkbank harbor-doctor` checks that cases can run and prints its findings as
JSON, with exit `0` when ready and `2` otherwise. It is read-only: it never
installs, starts or pulls anything. It does not look inside the image either: an
image given with `--image` that lacks the compiler shows at the first code case,
as `unsupported/sandbox_unavailable`.

| `harbor-doctor` reports | Fix |
|---|---|
| `Harbor requires Python >=3.12` | `./install.sh`; it creates `.venv` with Python 3.12 or newer. |
| `Install the hash-locked requirements-harbor.txt` or `Install the exact archive in requirements-harbor.txt` | `./install.sh` |
| `Docker CLI is unavailable` | Install Docker so that the `docker` command works for your user, then `./install.sh`. |
| `Docker access or the explicitly prepared image is missing` | `./install.sh`; it checks your access to Docker and pulls the pinned image. For an image given with `--image`: `docker pull IMAGE`. |
| `Docker image inspection failed` | Start Docker and check that `docker info` answers. |

When `install.sh` itself stops:

- **Before anything changed (exit `1`):** it lists what is missing. For Docker it
  tells apart a missing `docker` command, a user outside the `docker` group, a
  login session older than the group membership (log out and in again; under
  WSL2 run `wsl --shutdown` in Windows and open the distribution again) and a
  Docker that does not answer (`systemctl start docker`). For Python it names a
  missing interpreter, one older than 3.12 or a missing `venv` module (Debian and
  Ubuntu: `apt install python3-venv`).
- **At a step (exit `2`):** it names the failed step. Fix the cause and run
  `./install.sh` again; it continues where it stopped. If `.venv` itself is
  broken, remove it first (`rm -rf .venv`). A lock step fails without network
  access, or when a download does not match its pinned hash; a mismatch is a
  reason to stop, not to change the hash.

## Updating

Update the checkout (`git pull`), then run `./install.sh` again: it registers
new commands and modules, pulls a newly pinned image and repeats both checks.
`./werkbank` alone does not notice an update whose lock files are unchanged.
When the lock files changed, `install.sh` refuses the old `.venv`: remove it
(`rm -rf .venv`) and run the script again.

A new release archive is unpacked into a new folder and installed there.

Results are comparable only within one Werkbank version
([comparability](METHODOLOGY.md#comparability)), so finish a series before you
update.

## Removing

Deleting the Werkbank folder removes the installation and, under `runs/`, your
results. To keep the folder and remove only the installation:

```sh
rm -rf .venv .werkbank-tools src/victors_werkbank.egg-info
```

`.werkbank-tools/` exists only when Python was provided on demand. Outside the
folder there are the image in Docker's image store, pip's download cache and, if
you submitted results, the installation ID in
`~/.config/werkbank/installation-id` (under `$XDG_CONFIG_HOME` when that is set;
[submitting results](SUBMISSION.md) says what it is for):

```sh
docker image rm ghcr.io/aisyndicate-ch/werkbank-runtime@sha256:2c57f1598aa42b357d8220880591ca5c09939621ed5c12d12c05a1f6c0521f36
```
