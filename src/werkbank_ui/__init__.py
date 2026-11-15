"""The browser interface: the guided start and a view of werkbank run and
evaluation directories.

A separate package beside `victors_werkbank`, so it lies outside the harness
digest: a change here never splits pools. The view reads through the
package's own validation and rendering and writes nothing; the control side
runs the guided start of `werkbank_start`, every pass a `werkbank run` child
process, and hands a submission to `werkbank_submit`. It listens on loopback
only, behind Host, Origin and per-session token checks, and acts only on
forms that carry the session's form token (`docs/users/INTERFACE.md`).
"""
