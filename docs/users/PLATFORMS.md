# Platforms

Werkbank runs on Linux and, on a Windows host, inside WSL2. On a Mac it runs,
experimentally, inside a Linux virtual machine ([macOS](#macos-experimental)).
Native macOS and native Windows are not supported. Code that a model submits executes only in Linux
containers; nothing falls back to the host ([security](../../SECURITY.md)).

## Support matrix

Support is claimed per tested combination of operating system, architecture,
Python and container backend. A combination that is not listed is not supported.

| Host | Arch | Python | Containers | What is tested | How |
|---|---|---|---|---|---|
| Ubuntu 24.04 | amd64 | 3.12, 3.13 | none | Installation from the hash-locked requirements; every offline command; the browser interface; a clean wheel installation without Harbor | Continuous integration on GitHub-hosted `ubuntu-24.04` |
| Ubuntu 24.04 | amd64 | 3.12 | Docker Engine | `harbor-doctor` and `harbor-smoke`; cases as Harbor trials with regrade and export; the controls of the code containers | Continuous integration on GitHub-hosted `ubuntu-24.04` |
| Ubuntu 24.04 under WSL2 on a Windows host | amd64 | 3.12 | Docker Engine inside WSL2 | Installation with `install.sh`; `harbor-doctor` and `harbor-smoke`; cases as Harbor trials with regrade and export; `werkbank start` from the readiness check to the result page | One [recorded run](#wsl2-receipt) on the pinned image |
| Linux | arm64 | 3.12 | none | Installation of the Harbor requirements only; no case has run | [arm64](#arm64) |
| macOS on Apple silicon, Werkbank in a Linux VM | arm64 | 3.12 inside the VM | Docker Engine inside the VM | Nothing yet: experimental | [macOS](#macos-experimental) |
| macOS, native | any | any | any | Not supported; use a [Linux VM](#macos-experimental) | [Not supported](#not-supported-native-macos-and-windows) |
| Windows, native | any | any | any | Not supported; use [WSL2](#wsl2) | [Not supported](#not-supported-native-macos-and-windows) |

- "Continuous integration" is this repository's own workflow: on every pull
  request and on request it checks that the tree installs, starts and builds, and
  it runs Werkbank's test suite, the cases in containers included. The maintainers
  run the same suite on this tree before a release.
- Docker Desktop with its WSL2 backend is untested. The tested route is Docker
  Engine installed inside the WSL2 distribution.
- The Python column names the system Python a row was tested with. No row covers
  the Python that `install.sh` can provide on demand, on x86_64 and aarch64 Linux
  ([Python on demand](../INSTALLATION.md#python-on-demand)).

## arm64

The pinned runtime image is a multi-architecture index: one digest names a
linux/amd64 and a linux/arm64 image. On arm64 only the image's build is checked:
under emulation it compiles and runs one test program. No case, `harbor-smoke` or
container control has run on an arm64 host, so running cases there is not claimed.

## macOS (experimental)

On a Mac, Werkbank runs inside a Linux virtual machine (VM), with Docker Engine
inside that VM, and the model server runs natively on the Mac. Werkbank's files
and its containers stay on Linux, so the results are the same kind of results as on
any Linux computer. This route is **experimental**: it follows the public
documentation of the tools named below, and no run on a Mac has been recorded yet.
Until one is, every Mac combination counts as untested.

The recommended VM is [Lima](https://lima-vm.io/), which is free and open source
and gives the VM the Mac's address under a fixed name. The steps:

1. Install Lima on the Mac, for example with `brew install lima`.
2. Create and start an Ubuntu 24.04 VM, and open a shell in it:

   ```sh
   limactl start --name=werkbank --cpus=4 --memory=4 template:ubuntu-24.04
   limactl shell werkbank
   ```

   Lima's defaults are 4 CPUs, 4 GiB of memory and a 100 GiB disk. On Apple
   silicon the VM's memory comes out of the same unified memory the model uses, so
   give the VM what Docker, the [runtime image](../INSTALLATION.md#requirements)
   and a run's containers need, and leave the rest to the model.
3. Inside the VM, install Docker Engine following Docker's
   [instructions for Ubuntu](https://docs.docker.com/engine/install/ubuntu/), and
   add your user to the `docker` group. Restart the VM so that the group applies
   (`exit`, then `limactl stop werkbank`, `limactl start werkbank` and `limactl
   shell werkbank` again); `docker info` then answers.
4. Install `git` and `python3-venv` with `apt`, change to the VM's own home
   directory (`cd ~`) and clone the repository there. Keep it and every run
   directory in the VM's file system, never under `/Users`: Lima shares the Mac's
   home folder read-only at its own path, and a shared folder does not give the
   locking and `fsync` Werkbank's evidence files rely on.
5. Run `./werkbank` in the clone ([installation](../INSTALLATION.md)). The
   installation sees that it runs in a Lima VM and names the Mac's address.
6. The model server runs on the Mac, for example Ollama (port 11434), LM Studio
   (port 1234) or `mlx_lm.server` from mlx-lm (port 8080). From the VM, the Mac is
   `host.lima.internal`, not `127.0.0.1`, and Lima's documentation states that
   this name reaches servers that listen only on the Mac's `127.0.0.1` too. Give
   the base URL with that name, for example
   `--endpoint http://host.lima.internal:11434/v1`, or type it when the
   [guided start](CLI.md#guided-start) asks: its search looks only at `127.0.0.1`
   inside the VM, finds nothing on the Mac, and then names this address. A Lima VM
   created with `networkOpts.hostLoopback: false` cannot reach the Mac this way.
7. Nothing reads the Mac's hardware: the GPU read uses `nvidia-smi`, which a Mac
   does not have, and it only reads the machine Werkbank runs on. State the
   hardware with `--hardware-label`, for example `--hardware-label "Apple M4 Max
   128 GB"`, and, if you like, a [serving note](CLI.md#sampling-and-the-recorded-setup)
   with `"accelerator"`, `"accelerator_memory_gib"` and `"memory_type":
   "unified"`.
8. The [browser interface](INTERFACE.md) runs inside the VM. Lima forwards a port
   that listens on the VM's `127.0.0.1` to the Mac's `localhost`, so the browser
   on the Mac opens the printed address.

Ollama serves a short context unless it is told otherwise: start it with
`OLLAMA_CONTEXT_LENGTH` set to the context the mode needs, which
[`werkbank preflight`](CLI.md#check-the-server-first) names.

Alternatives:

- **Colima**, which is built on Lima: its VM already runs Docker. Open a shell in
  it with `colima ssh` and continue with step 4; the Mac is `host.lima.internal`
  there as well. Colima's default VM has 2 CPUs and 2 GiB of memory; give it more
  with `colima start --cpu 4 --memory 4`.
- **OrbStack**: create an Ubuntu machine (`orb create ubuntu werkbank`) and
  continue with step 3. From a machine, the Mac is `host.orb.internal`;
  `127.0.0.1` there is the machine itself. OrbStack's documentation does not say
  whether a server that listens only on the Mac's `127.0.0.1` answers at that
  name, nor whether Docker Engine runs inside a machine. If the server does not
  answer, let it listen on all addresses (LM Studio: serve on the local network;
  Ollama: `OLLAMA_HOST=0.0.0.0`); other computers on your network can then reach it
  too. The Mac's files appear under `/mnt/mac`; keep the clone in the machine's own
  file system. OrbStack is free for personal use only.
- **Docker Desktop** alone is no route: it runs containers, but gives no Linux
  system to install Werkbank in.

Known limits:

- Nothing on a Mac is tested: no installation, no case, no `harbor-smoke`. The
  runtime image's linux/arm64 build is checked only under emulation
  ([arm64](#arm64)).
- The Mac's hardware is never read; only what you state with `--hardware-label`
  or a serving note is recorded.
- LM Studio and Ollama are recognized from their model list, the mlx-lm server
  is not ([a model served on another machine](#a-model-served-on-another-machine)).

## WSL2

The route for a Windows host, as the [recorded run](#wsl2-receipt) took it:

1. Install WSL2 with Ubuntu 24.04 (`wsl --install -d Ubuntu-24.04` in an
   administrator PowerShell) and enable systemd in the distribution's
   `/etc/wsl.conf`:

   ```ini
   [boot]
   systemd=true
   ```

   Restart WSL (`wsl --shutdown`) and open the distribution again. The memory and
   processor limits of the WSL2 VM live in `%UserProfile%\.wslconfig` on the
   Windows side. Docker runs inside that VM, so give it room for the containers.
2. Install Docker Engine inside the distribution, following Docker's
   [instructions for Ubuntu](https://docs.docker.com/engine/install/ubuntu/), and
   add your user to the `docker` group. `systemctl is-active docker` reports
   `active`.
3. Install `git` and `python3-venv` with `apt`. Without `python3-venv`, the
   installation offers to [provide Python](../INSTALLATION.md#python-on-demand)
   instead.
4. Clone the repository **inside the Linux file system**, for example under `~`,
   and run `./werkbank` in it ([installation](../INSTALLATION.md)). Keep the
   repository and every run directory there, never under `/mnt/c` or another
   Windows drive: those are reached over a network file system that is slow and
   whose locking and `fsync` Werkbank's evidence files do not rely on.
5. Optionally check the runtime with
   [`harbor-smoke`](../INSTALLATION.md#harbor-runtime-python-312-and-docker).
6. The model server:
   - A server elsewhere on the network is reached by its address.
   - A server on the Windows host is reached through the Windows host's address
     as WSL2 sees it (in the default NAT mode, the default gateway of `ip
     route`), or as `127.0.0.1` in WSL2's mirrored networking mode.
   - The [guided start](CLI.md#guided-start) searches only `127.0.0.1` as WSL2
     sees it. It finds a server on the Windows host only in mirrored mode; give
     any other server with `--endpoint`. When its search finds nothing under
     WSL2, it says so.
7. The [browser interface](INTERFACE.md) runs inside WSL2. The Windows browser
   opens the printed address, as `127.0.0.1` or `localhost`, because WSL2
   forwards loopback ports to Windows; this was checked in the default NAT mode.

## WSL2 receipt

The WSL2 row of the matrix rests on one recorded run, not on continuous
integration, whose hosted runners provide no WSL2.

- **Where:** Ubuntu 24.04.5 LTS with systemd under WSL2 on a Windows host (kernel
  6.18.54.1-microsoft-standard-WSL2), Python 3.12.3, Docker Engine 29.9.1 inside
  the distribution (not Docker Desktop), Harbor 0.24.0, this repository's tree in
  the Linux file system, and the pinned runtime image.
- **What passed:** `./install.sh` twice, into a new `.venv` and then reusing it;
  `werkbank doctor --offline`, `werkbank harbor-doctor` and `werkbank
  harbor-smoke`; cases as Harbor trials, host-simulated and code tasks, with an
  export and a regrade from it; `werkbank start --mode quick --yes` against a
  scripted stand-in for a model on a loopback port, from the readiness check
  through three passes to the result page; and the release archive, installed on
  its own.
- **What it does not cover:** a real model server, Docker Desktop, and any other
  distribution or Windows version than the one named.

## A model served on another machine

The platform rules apply to the computer that runs Werkbank. The model server may
run elsewhere, a Mac with Apple silicon included, with LM Studio, Ollama or the
mlx-lm server, as long as Werkbank reaches its address; for Werkbank itself on a
Mac, see [macOS](#macos-experimental). LM Studio and Ollama (for a model of its own
library) are recognized from their model list; the mlx-lm server is not. Nothing
about the hardware is read for such a server: state it with `--hardware-label` and
a [serving note](CLI.md#sampling-and-the-recorded-setup) (for a Mac: the chip, its
unified memory and `unified`).

## Not supported: native macOS and Windows

Werkbank's evidence files rely on Linux file semantics: `fcntl` locks,
`O_NOFOLLOW`, `O_NONBLOCK`, directory `fsync`, hard links and POSIX path
separators in digests. macOS and Windows lack some of these or guarantee less.
Supporting them would change the evidence layer and with it the harness digest.
On Windows, use [WSL2](#wsl2); on a Mac, a [Linux VM](#macos-experimental).
