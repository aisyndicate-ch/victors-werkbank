# The leaderboard

The leaderboard is at [https://aisyndicate.ch/werkbank-leaderboard/](https://aisyndicate.ch/werkbank-leaderboard/).

The leaderboard lists Werkbank results that testers submitted and the maintainer
checked again on their own. It ranks setups, not models, on the Full set,
separately for each Werkbank version and each hardware class. Taking part is
voluntary: a result appears only when you submit it and agree to its publication.

## What an entry is

An entry is one setup measured in three passes of the Full set. A setup is
everything that answered the tasks: the model and its quantization, the server
software and its settings, the hardware, the thinking setting sent, the context,
the per-response cap, sampling and streaming, and the Werkbank version with its
container image ([comparability](../METHODOLOGY.md#comparability)). The same model
with other settings or on other hardware is another entry.

- **Case set.** Only the Full set, every scored task, enters the leaderboard.
  Normal has an overall score with its interval on the result page but no
  leaderboard, and Quick has no score; scores on different sets never compare, so
  passes of Full never enter as Normal. The German module stands outside every
  score and so outside the leaderboard; a trial, a single pass and an own selection
  of tasks never enter it.
- **Versions.** Every Werkbank version has its own leaderboard, as results of
  different versions never compare. The leaderboard of an earlier version stays as
  it was.
- **Passes.** An entry needs three passes of one setup with no pass stopped by a
  technical problem, the same as a score on the result page
  ([repetitions](../METHODOLOGY.md#sampling-repetitions-and-uncertainty)).
- **One entry per submission.** Each submission gives its own entry; submissions
  are never pooled with each other. A newer submission of the same setup from the
  same installation replaces the older one. The same setup submitted from
  different installations stands in rows of its own, marked as one setup, which
  shows how well a result repeats.
- **The maintainer's runs.** The maintainer's own measurement runs of a version
  enter under the same rules, as ordinary entries named "Werkbank maintainer".

## What a submission needs

A submission for the leaderboard is an ordinary submission
([submitting results](../SUBMISSION.md)) with two answers more:

- **Raw data included.** The maintainer checks every entry by grading your runs
  again, which needs each run's export archive. Without raw data a submission
  cannot enter the leaderboard.
- **Consent to publication.** It is asked apart from the consent to send, and you
  can give a display name.

Nothing else is required: no serving note and no hand-written setting. Whatever
Werkbank recorded about your setup is shown, and a setting it could not record is
shown as "unknown" and keeps no entry out.

## How an entry is checked

Every number on the leaderboard comes from the maintainer's own grading, never from
the submitted figures. The maintainer grades every run of a submission again from
its raw data with the Werkbank build that wrote it, in a disposable isolated
environment that trusts nothing in the archive, and evaluates the graded runs with
that build. A submission does not enter the leaderboard when:

- grading it again fails, or gives any attempt another status than the submitted
  one;
- its evaluation differs from the submitted one;
- it was written by a build that is not a released Werkbank version.

It then stays an ordinary submission, kept and deleted as any other.

## Hardware classes

Each version has one table per hardware class and an overview of all entries. The class
comes from the accelerator memory Werkbank recorded for the setup, in whole GiB: the
memory stated in a serving note, otherwise the memory of the GPUs Werkbank read on
this computer, added up over all of them.

| Class | Accelerator memory |
| --- | --- |
| up to 24 GiB | 24 GiB or less |
| 24 to 48 GiB | more than 24, up to 48 GiB |
| 48 to 96 GiB | more than 48, up to 96 GiB |
| 96 to 128 GiB | more than 96, up to 128 GiB |
| over 128 GiB | more than 128 GiB |

- A card sold as 24 GB records 24 GiB and stands in the first class; two such
  cards record 48 GiB and stand in the second.
- Unified memory, shared by the CPU and the GPU as on a Mac, counts in full, the
  whole memory of the machine, and stands in the same classes as GPU memory; the
  memory type is shown beside the class. A Mac states it in its serving note
  ([recorded setup](CLI.md#sampling-and-the-recorded-setup)), as Werkbank reads no
  hardware there.
- A hardware label is shown as you wrote it and never sets the class.
- A setup without recorded memory, such as a server on another machine without a
  serving note, is in no class. It stands only in the overview, as "hardware not
  stated".
- The maintainer may refuse an entry whose memory plainly contradicts the
  accelerator it names.

## Reading the leaderboard

Entries rank by their overall score on the Full set
([scoring](../METHODOLOGY.md#scoring)). Two entries are separable when the interval
of their paired difference leaves out 0, or, where there is no paired difference,
when their own intervals do not overlap
([paired comparisons](../METHODOLOGY.md#pooling-paired-comparisons-and-the-pareto-view)).
An entry's rank is one more than the number of entries with a higher score that are
separable from it, so entries that cannot be told apart share a rank.

Each entry shows:

- the overall score with its 95 % interval;
- the solved rate of each category;
- the security line: the security boundaries the setup broke
  ([security findings](../METHODOLOGY.md#security-findings));
- the censored attempts, those that ended at a budget or by a technical problem,
  which count as not solved ([censoring](../METHODOLOGY.md#censoring));
- the tasks solved per pass;
- the setup: model, quantization, server software and version, hardware statement,
  hardware class and memory type, thinking setting sent, context, per-response cap,
  sampling and streaming;
- the Werkbank version, the display name and the date of the submission.

Seconds and tokens per solved task are shown for information and rank nothing:
they depend on the hardware even within a class. Everything a result page cannot
tell you, a leaderboard entry cannot either
([what results show](LIMITS.md)): an entry says how often one setup solved the Full
set's tasks under this version, not how good a model is in general.

## Display name

An entry shows the display name you gave, or "anonymous". A name is optional, and a
pseudonym is welcome; there is no account, so a name belongs to no one. It has 1 to
32 characters: letters, digits, spaces, `-`, `_` and `.`, beginning and ending with
a letter or a digit, with no web address and no `@`. Before an entry is published
the maintainer reads the name and replaces an insulting one, or one that pretends to
be another person or organization, by "anonymous"; the entry itself stays.

## Withdrawing an entry

Write to the deletion contact that `werkbank submit` shows and quote your
installation ID ([retention and deletion](../SUBMISSION.md#retention-and-deletion)).
You can ask for either:

- **leaving the leaderboard:** every entry of your installation ID is removed, and
  your submissions stay with the maintainer;
- **deletion:** every entry is removed and your submissions are deleted.

A withdrawal takes effect with the next update of the leaderboard, within 30 days
at the latest. The maintainer may remove an entry for abuse.

A published entry holds only the values the leaderboard shows. It stays until you
withdraw it: when your submission is deleted after its retention period, its raw
data go with it, and the entry stays.
