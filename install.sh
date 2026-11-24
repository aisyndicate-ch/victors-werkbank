#!/usr/bin/env bash
# One-step installation of Victor's Werkbank into .venv beside this script.
# It checks the prerequisites, runs the hash-locked steps of
# docs/INSTALLATION.md#by-hand, pulls the pinned container image by its digest and
# runs the read-only checks. It asks one question, and only at a terminal: whether to
# provide Python inside this checkout when no usable one is found. It never asks
# for administrator rights, writes files only inside this checkout apart from
# pip's download cache and Docker's image store, and downloads only what a lock
# file, the image digest in the package or this script pins.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "install.sh: run it with bash: ./install.sh" >&2
  exit 1
fi
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install.sh [--skip-docker | --check]

Creates .venv in this checkout, installs the hash-locked requirements and the
package, pulls the pinned container image and runs `werkbank doctor --offline`
and `werkbank harbor-doctor`. It is safe to run again.

  --skip-docker  install the Python environment only: no Docker check, no image
                 pull and no harbor-doctor; cases cannot run until the script
                 runs again without it
  --check        change and print nothing: exit 0 when a complete run without
                 --skip-docker (image pulled, checks passed) installed .venv
                 from this checkout's lock files, else 1 (./werkbank asks this)
  -h, --help     show this help

The interpreter is $WERKBANK_PYTHON if set, else the first of python3, python3.13
and python3.12 that is Python 3.12 or newer with its venv module. When there is
none and WERKBANK_PYTHON is not set, it asks at a terminal (default no) whether
to provide Python 3.12 inside this checkout, in .werkbank-tools/, and first says
what it downloads from where; the download is checked against a SHA-256 pinned
in this script. Without a terminal it stops as for any missing prerequisite.

Exit codes: 0 ready; 1 a prerequisite is missing or the existing .venv is
refused, and nothing was changed; 2 a step failed: fix the cause it shows and
run the script again, it continues where it stopped.
EOF
}

say() { printf 'install.sh: %s\n' "$*"; }

skip_docker=0
check=0
for argument in "$@"; do
  case "$argument" in
    --skip-docker) skip_docker=1 ;;
    --check) check=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
done

# macOS itself is no Linux: Werkbank runs in a Linux VM on the Mac instead.
if [[ "$(uname -s)" == Darwin ]]; then
  if ((check)); then
    exit 1
  fi
  echo "install.sh: this is macOS, where Werkbank does not run natively. Nothing" \
    "was changed. Run it inside a Linux VM on this Mac, with the model server on" \
    "the Mac (experimental): docs/users/PLATFORMS.md#macos-experimental" >&2
  exit 1
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$root"
LOCKS=(requirements-dev.txt requirements-harbor-build.txt requirements-harbor.txt)
for file in pyproject.toml "${LOCKS[@]}"; do
  if [[ ! -f "$file" ]]; then
    if ((check)); then
      exit 1
    fi
    echo "install.sh: $file is missing; run the script from a complete checkout." >&2
    exit 1
  fi
done

STAMP=.venv/werkbank-install
STAMP_FORMAT=werkbank-install-v1
digest=$(sha256sum "${LOCKS[@]}")

# new: no .venv; installing: created here, steps not yet complete; installed:
# the lock steps completed from these lock files; other: installed from other
# lock files; foreign: not created by this script.
venv_state() {
  if [[ ! -e .venv && ! -L .venv ]]; then
    echo new
  elif [[ -L .venv || ! -d .venv || -L "$STAMP" || ! -f "$STAMP" ]]; then
    echo foreign
  elif [[ "$(tail -n +2 "$STAMP")" != "$digest" ]]; then
    echo other
  else
    case "$(head -n 1 "$STAMP")" in
      "$STAMP_FORMAT installed") echo installed ;;
      "$STAMP_FORMAT installing") echo installing ;;
      *) echo foreign ;;
    esac
  fi
}

write_stamp() {
  printf '%s %s\n%s\n' "$STAMP_FORMAT" "$1" "$digest" >"$STAMP.tmp"
  mv -f -- "$STAMP.tmp" "$STAMP"
}

# Written as the last step of a complete run without --skip-docker and removed
# when a run starts changing anything, so it vouches for the whole
# installation: the lock steps, the package, the image and both checks. The
# stamp above only records the lock steps, so that a run can resume them.
READY=.venv/werkbank-ready
READY_FORMAT=werkbank-ready-v1
ready() {
  [[ -f "$READY" && ! -L "$READY" ]] &&
    [[ "$(cat -- "$READY")" == "$READY_FORMAT"$'\n'"$digest" ]]
}

# Python on demand: uv, pinned by its version and by the SHA-256 of its
# release archive for each machine, installs this CPython into TOOLS_DIR inside
# the checkout. uv checks the Python archive against the checksums built into it.
UV_VERSION=0.13.5
UV_SHA256_x86_64=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f
UV_SHA256_aarch64=4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403
PROVIDED_PYTHON=3.12.15
TOOLS_DIR=.werkbank-tools
case "$(uname -m)" in
  x86_64 | amd64) machine=x86_64 ;;
  aarch64 | arm64) machine=aarch64 ;;
  *) machine= ;;
esac
provided="$root/$TOOLS_DIR/python/cpython-$PROVIDED_PYTHON-linux-$machine-gnu/bin/python3.12"

# The interpreter that creates .venv: $WERKBANK_PYTHON, else the first
# candidate on PATH that is Python 3.12 or newer with its venv module, else the
# one an earlier run provided, else the first candidate that is 3.12 or newer
# (whose missing venv module is then named).
python_ok() {
  "$1" -c 'import sys; sys.exit(sys.version_info < (3, 12))' >/dev/null 2>&1
}
python_usable() {
  python_ok "$1" && "$1" -m ensurepip --version >/dev/null 2>&1
}
choose_python() {
  local candidate
  if [[ -n "${WERKBANK_PYTHON:-}" ]]; then
    printf '%s\n' "$WERKBANK_PYTHON"
    return
  fi
  for candidate in python3 python3.13 python3.12 "$provided"; do
    if command -v "$candidate" >/dev/null 2>&1 && python_usable "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  for candidate in python3 python3.13 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf 'python3\n'
}

# Downloads the HTTPS URL $1 to $2 with the urllib of any python3, else with
# curl; neither keeps anything that left HTTPS. The caller checks the hash.
fetch() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -I -c '
import shutil, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=60) as response:
    if not response.geturl().startswith("https://"):
        sys.exit("the download left HTTPS")
    with open(sys.argv[2], "wb") as file:
        shutil.copyfileobj(response, file)
' "$1" "$2"
  else
    curl --proto =https --proto-redir =https --fail --silent --show-error \
      --location --connect-timeout 60 --output "$2" "$1"
  fi
}

# Whether this run may offer to provide Python: no interpreter was named, the
# machine has a pinned uv, something can download, and a person can answer.
can_provide() {
  [[ -z "${WERKBANK_PYTHON:-}" && -n "$machine" && -t 0 && -t 1 ]] &&
    { command -v python3 || command -v curl; } >/dev/null 2>&1
}

uv_archive() { printf 'uv-%s-unknown-linux-gnu.tar.gz\n' "$machine"; }
uv_url() {
  printf 'https://github.com/astral-sh/uv/releases/download/%s/%s\n' \
    "$UV_VERSION" "$(uv_archive)"
}

# Says exactly what providing Python loads from where; true on y or yes.
offer_python() {
  local answer
  cat <<TEXT
install.sh: no Python 3.12 or newer with its venv module was found.
It can provide one inside this checkout, in $TOOLS_DIR/, without administrator
rights and without changing anything outside the checkout:
  1. download uv $UV_VERSION from
     $(uv_url)
     and check it against the SHA-256 pinned in install.sh before unpacking it;
  2. have uv install CPython $PROVIDED_PYTHON from the releases of
     https://github.com/astral-sh/python-build-standalone, which uv checks
     against the SHA-256 values built into it.
.venv is then created with that Python. Only the installation downloads; runs
stay offline. To undo it, remove $TOOLS_DIR/ and .venv.
TEXT
  read -r -p "Provide Python $PROVIDED_PYTHON this way? [y/N] " answer || return 1
  [[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]]
}

# Inside a Linux VM, the computer that runs it is not 127.0.0.1. Lima, and
# Colima built on it, write host.lima.internal into the guest's /etc/hosts at
# every boot; an OrbStack machine shows the Mac's files at /mnt/mac. Both are
# only read, never contacted.
HOSTS=/etc/hosts
MAC_FILES=/mnt/mac
newline=$'\n'
lima_line="${newline}[0-9A-Fa-f.:]+[[:blank:]]+host\\.lima\\.internal([[:blank:]]|${newline})"
vm=
if [[ -f "$HOSTS" && -r "$HOSTS" ]] &&
  [[ "${newline}$(<"$HOSTS")${newline}" =~ $lima_line ]]; then
  vm=lima
elif [[ -d "$MAC_FILES" ]]; then
  vm=orbstack
fi
docker_guide="under WSL2 see docs/users/PLATFORMS.md#wsl2"
relogin="under WSL2: run wsl --shutdown in Windows and open the distribution again"
case "$vm" in
  lima)
    docker_guide="in this VM see docs/users/PLATFORMS.md#macos-experimental"
    relogin="in a Lima VM: limactl stop and limactl start it"
    ;;
  orbstack)
    docker_guide="in this VM see docs/users/PLATFORMS.md#macos-experimental"
    relogin="in an OrbStack machine: restart the machine"
    ;;
esac

# Prints what keeps Docker from working for this user; nothing when it answers.
docker_problem() {
  local output user
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed: install Docker Engine so that the docker command" \
      "works for your user ($docker_guide)."
    return
  fi
  if output=$(timeout 30 docker info --format '{{.ServerVersion}}' 2>&1); then
    return
  fi
  if [[ "$output" == *"ermission denied"* ]]; then
    user=$(id -un)
    if [[ " $(id -nG "$user") " != *" docker "* ]]; then
      echo "Docker refuses your user: add it to the docker group as administrator" \
        "(usermod -aG docker $user), then log out and in again ($relogin)."
    elif [[ " $(id -nG) " != *" docker "* ]]; then
      echo "Docker refuses your user: it is in the docker group, but this login" \
        "session started before that; log out and in again ($relogin)."
    else
      echo "Docker refuses your user although it is in the docker group; see what" \
        "docker info reports."
    fi
  else
    echo "Docker does not answer: start it (with systemd, as administrator:" \
      "systemctl start docker) and check that docker info works."
  fi
}

state=$(venv_state)
if ((check)); then
  [[ "$state" == installed ]] && ready && exit 0
  exit 1
fi
problems=()
case "$state" in
  foreign)
    problems+=(".venv exists, but this script did not create it, so it cannot tell
  what it holds. Keep it and install by hand (docs/INSTALLATION.md#by-hand), or remove it
  (rm -rf .venv) and run ./install.sh again.")
    ;;
  other)
    problems+=(".venv was installed from other lock files than this checkout's.
  Remove it (rm -rf .venv) and run ./install.sh again.")
    ;;
esac

interpreter=
python_problem=
if [[ "$state" == new ]]; then
  interpreter=$(choose_python)
  if ! command -v "$interpreter" >/dev/null 2>&1; then
    python_problem="Python 3.12 or newer is not installed (looked for python3,
  python3.13 and python3.12): install it, or name it with WERKBANK_PYTHON."
  elif ! python_ok "$interpreter"; then
    python_problem="$interpreter is older than Python 3.12: install Python 3.12 or newer, or
  name it with WERKBANK_PYTHON."
  elif ! "$interpreter" -m ensurepip --version >/dev/null 2>&1; then
    python_problem="The venv module of $interpreter cannot install pip: install it as
  administrator (Debian and Ubuntu: apt install python3-venv, or the package for
  this version, such as python3.12-venv)."
  fi
fi

if ((!skip_docker)); then
  problem=$(docker_problem)
  if [[ -n "$problem" ]]; then
    problems+=("$problem")
  fi
fi

# Only a git checkout is updated with git; an unpacked release archive is not.
if [[ -e .git ]] && ! command -v git >/dev/null 2>&1; then
  say "note: git is not installed. This installation does not need it; cloning" \
    "and updating the repository do."
fi

case "$vm" in
  lima)
    say "note: this is a Lima VM (Lima or Colima). A model server on the computer" \
      "that runs it is reached as host.lima.internal, not 127.0.0.1, for example" \
      "http://host.lima.internal:11434/v1 (docs/users/PLATFORMS.md#macos-experimental)."
    ;;
  orbstack)
    say "note: this looks like an OrbStack machine. A model server on the Mac is" \
      "reached as host.orb.internal, not 127.0.0.1, for example" \
      "http://host.orb.internal:11434/v1 (docs/users/PLATFORMS.md#macos-experimental)."
    ;;
esac
if [[ -n "$vm" && ( "$root" == /Users/* || "$root" == "$MAC_FILES"/* ) ]]; then
  say "note: this checkout lies in a folder shared from the Mac. Clone it into the" \
    "VM's own file system, for example under ~, and run it there: a shared folder" \
    "does not give the locking and fsync Werkbank's evidence files rely on."
fi

# Providing Python is offered only when it is the one problem left, so that
# nothing is downloaded for an installation that would stop anyway.
provide=0
if [[ -n "$python_problem" ]]; then
  if ((!${#problems[@]})) && can_provide; then
    if offer_python; then
      provide=1
    fi
  elif can_provide; then
    python_problem+="
  Once the other problems are fixed, ./install.sh offers to provide one."
  elif [[ -z "${WERKBANK_PYTHON:-}" && -n "$machine" ]]; then
    python_problem+="
  Run at a terminal, ./install.sh offers to provide one inside this checkout."
  fi
  if ((!provide)); then
    problems=("$python_problem" "${problems[@]}")
  fi
fi

if ((${#problems[@]})); then
  echo "install.sh: nothing was changed. Fix these and run ./install.sh again:" >&2
  for problem in "${problems[@]}"; do
    printf -- '- %s\n' "$problem" >&2
  done
  exit 1
fi

# One step, shown before it runs; a failure stops with exit 2.
step() {
  printf '+ %s\n' "$*"
  if ! "$@"; then
    echo "install.sh: this step failed: $*" >&2
    echo "  Fix the cause shown above and run ./install.sh again; it continues" \
      "where it stopped. If .venv itself is broken, remove it (rm -rf .venv) first." >&2
    exit 2
  fi
}

# Loads the pinned uv, checks its hash before anything of it is used, and has
# it install the pinned Python into TOOLS_DIR; uv's settings files, mirrors and
# a cache outside the checkout are not used.
provide_python() {
  local archive part expected actual
  archive=$(uv_archive)
  part="$TOOLS_DIR/$archive.part"
  step mkdir -p -- "$TOOLS_DIR"
  step fetch "$(uv_url)" "$part"
  expected="UV_SHA256_$machine"
  actual=$(sha256sum -- "$part")
  if [[ "${actual%% *}" != "${!expected}" ]]; then
    rm -f -- "$part"
    echo "install.sh: $archive does not match the SHA-256 pinned in install.sh" \
      "(expected ${!expected}, got ${actual%% *}); it was removed unused." >&2
    exit 2
  fi
  step tar -xzf "$part" -C "$TOOLS_DIR"
  rm -f -- "$part"
  step env -u UV_PYTHON_INSTALL_MIRROR -u UV_PYPY_INSTALL_MIRROR \
    -u UV_PYTHON_DOWNLOADS_JSON_URL UV_NO_CONFIG=1 \
    UV_PYTHON_INSTALL_DIR="$root/$TOOLS_DIR/python" \
    UV_CACHE_DIR="$root/$TOOLS_DIR/cache" \
    "$TOOLS_DIR/uv-$machine-unknown-linux-gnu/uv" python install --no-bin "$PROVIDED_PYTHON"
  if ! python_usable "$provided"; then
    echo "install.sh: the provided Python $PROVIDED_PYTHON cannot create .venv;" \
      "remove $TOOLS_DIR/ and run ./install.sh again." >&2
    exit 2
  fi
  interpreter=$provided
}

if [[ "$state" != new ]]; then
  rm -f -- "$READY"
fi

if ((provide)); then
  provide_python
fi

if [[ "$state" == new ]]; then
  say "creating .venv with $interpreter ($("$interpreter" -c 'import sys; print("Python", sys.version.split()[0])'))"
  if ! "$interpreter" -m venv .venv; then
    # Only this run created .venv, so nothing of the user's is removed.
    rm -rf -- .venv
    echo "install.sh: creating .venv failed; nothing was kept. If the message above" \
      "names ensurepip or venv, install the venv module of $interpreter as" \
      "administrator (Debian and Ubuntu: apt install python3-venv)." >&2
    exit 2
  fi
  write_stamp installing
fi

# The steps run as in an activated environment: uv_build, which the Harbor
# install builds with, finds its pinned executable on PATH. `python` and
# `werkbank` are always the ones in .venv, never another on PATH, and the
# checks see only this checkout's installation.
export VIRTUAL_ENV="$root/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONHOME PYTHONPATH
python() { "$VIRTUAL_ENV/bin/python" "$@"; }
werkbank() { "$VIRTUAL_ENV/bin/werkbank" "$@"; }

if [[ "$state" == installed ]]; then
  say "reusing .venv: it was installed from these lock files"
else
  if [[ "$state" == installing ]]; then
    say "continuing the installation into .venv that stopped before"
  fi
  step python -m pip install --require-hashes --index-url https://pypi.org/simple -r requirements-dev.txt
  step python -m pip install --require-hashes -r requirements-harbor-build.txt
  step python -m pip install --no-build-isolation --require-hashes -r requirements-harbor.txt
  write_stamp installed
fi
# The package itself installs every time: a new checkout may add commands or
# modules that the editable installation has to register.
step python -m pip install --no-deps --no-build-isolation -e .
step werkbank doctor --offline

if ((skip_docker)); then
  say "the Python environment is ready. Docker steps skipped: run ./install.sh" \
    "again without --skip-docker before running cases."
  exit 0
fi

# The image `werkbank harbor-doctor` checks by default, pinned by its digest.
pinned='^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$'
if ! image=$(python -c 'from victors_werkbank.harbor_smoke import SMOKE_IMAGE; print(SMOKE_IMAGE)') ||
  [[ ! "$image" =~ $pinned ]]; then
  echo "install.sh: the package names no image pinned by a digest; nothing pulled." >&2
  exit 2
fi
step docker pull "$image"
step werkbank harbor-doctor
printf '%s\n%s\n' "$READY_FORMAT" "$digest" >"$READY.tmp"
mv -f -- "$READY.tmp" "$READY"
say "ready. Next: ./werkbank, which runs .venv/bin/werkbank start" \
  "(README.md#guided-start); it contacts your model server."
