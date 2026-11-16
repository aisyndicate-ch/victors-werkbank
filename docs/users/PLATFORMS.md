# Platforms

Werkbank runs on Linux and, on a Windows host, inside WSL2. macOS and native
Windows are not supported. Code that a model submits executes only in Linux
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
| macOS | any | any | any | Not supported | [Not supported](#not-supported-macos-and-native-windows) |
| Windows, native | any | any | any | Not supported; use [WSL2](#wsl2) | [Not supported](#not-supported-macos-and-native-windows) |

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

## Not supported: macOS and native Windows

Werkbank's evidence files rely on Linux file semantics: `fcntl` locks,
`O_NOFOLLOW`, `O_NONBLOCK`, directory `fsync`, hard links and POSIX path
separators in digests. macOS and Windows lack some of these or guarantee less.
Supporting them would change the evidence layer and with it the harness digest.
On Windows, use [WSL2](#wsl2).
