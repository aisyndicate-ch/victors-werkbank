# The runtime image for code that needs a compiler: the pinned Python image, the
# Rust compiler and what its linker needs.
#
# Both bases are multi-architecture indexes, so each digest fixes linux/amd64 and
# linux/arm64 alike. This file has no syntax directive, which would fetch a
# frontend image by a mutable tag.

# rust:1.99.1-slim-bookworm, used for its toolchain only.
FROM rust@sha256:f2c004e3d2d8b547e8cf7ffbc29205385f4e71d23e53e4e383065db5e9f822cd AS toolchain

# Only the compiler, its libraries and this architecture's standard library. Without
# cargo no build script runs, and without rust-lld the linker's threads and address
# space do not grow with the host's CPUs. The official toolchain's files are
# world-writable: a fresh user could change the compiler for later executions in the
# same container, and the controller closes only directories.
RUN <<'EOF'
set -eu
sysroot="$(rustc --print sysroot)"
target="$(rustc --print host-tuple)"
mkdir -p /opt/rust/bin "/opt/rust/lib/rustlib/$target"
cp -a "$sysroot/bin/rustc" /opt/rust/bin/
cp -a "$sysroot"/lib/librustc_driver-*.so "$sysroot"/lib/libLLVM.so.* /opt/rust/lib/
cp -a "$sysroot/lib/rustlib/$target/lib" "/opt/rust/lib/rustlib/$target/"
chmod -R go-w /opt/rust
EOF

# python:3.12.15-slim-trixie. /usr/local/bin/python3 and setpriv stay where the
# controller and the task packages expect them.
FROM python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# The Debian archive as of the base image's own build, named in its sources file.
# The snapshot fixes every package and dependency for both architectures, so a
# rebuild installs the same versions and apt upgrades nothing in the base. The
# archive keys still verify the signatures; only the expiry of the old release files
# is not checked.
COPY <<EOF /etc/apt/sources.list.d/debian.sources
Types: deb
URIs: https://snapshot.debian.org/archive/debian/20261005T000000Z
Suites: trixie trixie-updates
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.pgp
Check-Valid-Until: no

Types: deb
URIs: https://snapshot.debian.org/archive/debian-security/20261005T000000Z
Suites: trixie-security
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.pgp
Check-Valid-Until: no
EOF

# rustc links through cc, and the C runtime files come with libc6-dev. The snapshot
# service refuses bursts with 503, so apt backs off and retries.
RUN <<'EOF'
set -eu
apt-get -o Acquire::Retries=10 update
apt-get -o Acquire::Retries=10 install -y --no-install-recommends gcc libc6-dev
rm -rf /var/lib/apt/lists/*
EOF

COPY --from=toolchain /opt/rust /opt/rust

# A failed build pushes nothing: every architecture builds and runs a test binary
# through cc and GNU ld. rustc accepts `-C linker-features=-lld` only on x86-64,
# whose default is rust-lld; the other targets link through cc anyway.
RUN <<'EOF'
set -eu
ln -s /opt/rust/bin/rustc /usr/local/bin/rustc
case "$(uname -m)" in
    x86_64) linker='-C linker-features=-lld' ;;
    *) linker= ;;
esac
dir="$(mktemp -d)"
cat > "$dir/smoke.rs" <<'RUST'
#[test]
fn adds() {
    assert_eq!(2 + 2, 4);
}
RUST
rustc --edition 2021 --test -C codegen-units=1 $linker -o "$dir/smoke" "$dir/smoke.rs"
"$dir/smoke"
rm -rf "$dir"
EOF
