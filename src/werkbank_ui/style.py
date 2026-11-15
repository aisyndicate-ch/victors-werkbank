"""The interface's only stylesheet, served from the interface itself.

Dark throughout, in the result page's own dark palette, so a result page
shown inside the interface reads as part of it. Plain surfaces, hairlines and
tabular figures: no gradients, shadows or decoration. A status is always a
word; its color only repeats it."""

STYLE = """
:root {
  color-scheme: dark;
  --ink: #f0efec; --muted: #c3c2b7; --dim: #8f8e86;
  --surface: #1a1a19; --raised: #212120; --note: #262624; --line: #45443f;
  --accent: #6ea8ec; --solved: #6ea8ec; --model: #e98a63; --budget: #4fc39a;
  --technical: #e0a836; --notrun: #a3a29a;
  --sans: system-ui, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, "DejaVu Sans Mono",
    monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.5 var(--sans); font-variant-numeric: tabular-nums; }
a { color: var(--accent); text-underline-offset: .15em; }
a:hover { color: var(--ink); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: .25rem 1.5rem;
  padding: .7rem 1.25rem; border-bottom: 1px solid var(--line); color: var(--dim); }
header a { text-decoration: none; }
header .brand { color: var(--ink); font-weight: 600; letter-spacing: .04em; }
header nav { display: flex; gap: 1.25rem; }
header .where { margin-left: auto; font-size: .85rem; }
main { padding: 1.25rem; max-width: 72rem; }
h1, h2, h3 { font-weight: 600; line-height: 1.25; overflow-wrap: anywhere; }
h1 { font-size: 1.45rem; margin: .25rem 0 .75rem; }
h2 { font-size: .8rem; margin: 2.25rem 0 .6rem; padding-bottom: .3rem;
  border-bottom: 1px solid var(--line); color: var(--muted);
  letter-spacing: .09em; text-transform: uppercase; }
h3 { font-size: 1rem; }
h1 code { font-size: .9em; }
p { margin: .5rem 0; max-width: 46rem; }
code, pre, input[type=text], select { font: 13px/1.45 var(--mono); }
code, pre { overflow-wrap: anywhere; }
pre { white-space: pre-wrap; margin: .6rem 0; }
.lead { color: var(--muted); }
.note, .hint { color: var(--dim); font-size: .9rem; }
.hint { margin: .2rem 0 0; }
ul.plain { padding-left: 1.1rem; max-width: 46rem; color: var(--muted); }
ul.plain li { margin: .3rem 0; }
.table { overflow-x: auto; }
table { border-collapse: collapse; margin: .6rem 0; }
th, td { border-bottom: 1px solid var(--line); padding: .3rem .9rem .3rem 0;
  text-align: left; vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: .85rem;
  border-bottom-color: var(--dim); }
nav.files { margin: .6rem 0 1rem; }
.badge { border: 1px solid var(--line); border-radius: 2px; padding: 0 .45rem;
  font-weight: 600; font-size: .85rem; letter-spacing: .03em; }
.badge.complete, .badge.running { border-color: var(--solved); color: var(--solved); }
.badge.interrupted, .badge.another-build, .badge.stopping, .badge.stopped,
.badge.evaluated, .badge.waiting-for-you { border-color: var(--technical);
  color: var(--technical); }
.badge.invalid, .badge.failed { border-color: var(--model); color: var(--model); }
.state { color: var(--muted); }
.panel, .question { border: 1px solid var(--line); background: var(--raised);
  padding: 1rem 1.25rem; max-width: 46rem; }
.question { border-left: 3px solid var(--accent); margin: 1.25rem 0; max-width: 60rem; }
.question h2 { margin-top: 0; border: 0; padding: 0; color: var(--accent); }
.context, .log { background: var(--note); border: 1px solid var(--line);
  padding: .75rem 1rem; max-height: 28rem; overflow: auto; }
.log { max-height: 36rem; overflow-anchor: none; }
.log .run { color: var(--dim); }
.log .error { color: var(--model); }
.log .answer { color: var(--accent); }
.prompt { font-weight: 600; }
.problem { border-left: 3px solid var(--model); padding: .4rem .9rem;
  background: var(--note); }
.field { margin: 0 0 .9rem; }
.field > label { display: block; font-weight: 600; margin-bottom: .2rem; }
input[type=text], select { width: 100%; max-width: 34rem; padding: .4rem .55rem;
  background: var(--surface); color: var(--ink); border: 1px solid var(--line);
  border-radius: 2px; }
select { width: auto; }
input[type=text]:focus, select:focus { border-color: var(--accent); outline: 0; }
details { margin: .75rem 0; }
summary { cursor: pointer; color: var(--muted); }
details[open] > summary { margin-bottom: .75rem; }
fieldset { border: 0; margin: 0; padding: 0; min-width: 0; }
legend { font-weight: 600; padding: 0; margin-bottom: .5rem; }
.choice { display: flex; gap: .6rem; align-items: baseline; padding: .45rem 0;
  border-top: 1px solid var(--line); cursor: pointer; }
.choice .detail { display: block; color: var(--muted); font-size: .9rem; }
button { font: 600 14px/1.2 var(--sans); padding: .5rem 1.1rem; cursor: pointer;
  background: transparent; color: var(--ink); border: 1px solid var(--dim);
  border-radius: 2px; }
button:hover { border-color: var(--ink); }
button.primary { background: var(--accent); border-color: var(--accent);
  color: #10161d; }
button.primary:hover { background: var(--ink); border-color: var(--ink); }
button.danger { border-color: var(--model); color: var(--model); }
form.stop { margin: 1.5rem 0; }
ol.steps { list-style: none; display: flex; flex-wrap: wrap; margin: 1rem 0;
  padding: 0; counter-reset: step; border: 1px solid var(--line); max-width: 60rem; }
ol.steps li { counter-increment: step; flex: 1 1 6.5rem; padding: .45rem .7rem;
  border-right: 1px solid var(--line); color: var(--dim); font-size: .85rem; }
ol.steps li:last-child { border-right: 0; }
ol.steps li::before { content: counter(step) " "; font-family: var(--mono); }
ol.steps li span { display: block; font-size: .75rem; letter-spacing: .06em;
  text-transform: uppercase; }
ol.steps li.done { color: var(--muted); }
ol.steps li.current { color: var(--ink); background: var(--note);
  box-shadow: inset 0 -2px 0 var(--accent); }
td.status { font-weight: 600; }
td.status.passed { color: var(--solved); }
td.status.failed { color: var(--model); }
td.status.timeout, td.status.output-limit { color: var(--budget); }
td.status.invalid, td.status.unsupported { color: var(--technical); }
td.status.not-run, td.status.running { color: var(--notrun); }
iframe.result { display: block; width: 100%; max-width: 60rem; height: 80vh;
  border: 1px solid var(--line); background: var(--surface); }
"""
