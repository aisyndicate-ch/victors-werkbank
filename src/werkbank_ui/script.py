"""The interface's only script, served from the interface itself.

It keeps a running series page current in place: every few seconds it reads
the same page again and puts what changed into the parts the page marks, so
nothing jumps while the log is read. It acts nowhere: it sends no form and
reads nothing but the series page of its own origin. Without scripts the
page loads itself again instead, and everything else works the same."""

LIVE = """"use strict";
(function () {
  var root = document.getElementById("series");
  if (!root || !root.hasAttribute("data-live")) { return; }
  var wait = (Number(root.getAttribute("data-live")) || 5) * 1000;
  var each = Array.prototype.forEach;

  function parts(scope) { return scope.querySelectorAll("[data-part]"); }
  function names(scope) {
    var found = [];
    each.call(parts(scope), function (part) {
      found.push(part.getAttribute("data-part"));
    });
    return found.join(" ");
  }
  function log(scope) { return scope.querySelector("pre.log"); }
  // Where a line stands inside the log's own frame.
  function within(pre, line) {
    return line.getBoundingClientRect().top - pre.getBoundingClientRect().top;
  }
  function atEnd(pre) {
    return pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  }

  // What the server last sent for each part: a part is touched only when
  // that changes, so what the reader opened or scrolled stays.
  var last = {};
  each.call(parts(root), function (part) {
    last[part.getAttribute("data-part")] = part.innerHTML;
  });
  var shown = log(root);
  var lastLog = shown ? shown.innerHTML : "";
  if (shown) { shown.scrollTop = shown.scrollHeight; }

  // Where the log the server sent last and the one it sends now overlap:
  // the place in the last one from which it is the beginning of the new
  // one, at a line's start; -1 without an overlap.
  function overlap(before, now) {
    var at = 0;
    while (at >= 0 && at < before.length) {
      if (now.indexOf(before.slice(at)) === 0) { return at; }
      at = before.indexOf("\\n", at);
      if (at >= 0) { at += 1; }
    }
    return -1;
  }

  // Everything of the part but the log itself, as the server sends it now.
  function around(old, fresh, pre, next) {
    while (old.firstChild && old.firstChild !== pre) {
      old.removeChild(old.firstChild);
    }
    var node = fresh.firstChild;
    while (node && node !== next) {
      old.insertBefore(document.importNode(node, true), pre);
      node = node.nextSibling;
    }
  }

  function updateLog(old, fresh) {
    var pre = log(old), next = log(fresh);
    var follow = pre ? atEnd(pre) : true;
    var text = next ? next.innerHTML : "";
    var at = pre && next && lastLog ? overlap(lastLog, text) : -1;
    if (at >= 0) {
      // The lines the server no longer shows leave at the top, the new ones
      // are added below, and the reader's place and selection stay: the
      // view moves by exactly what the first line that stays moved.
      var gone = lastLog.slice(0, at).split("\\n").length - 1;
      var stays = pre.querySelectorAll("span")[gone];
      var was = stays ? within(pre, stays) : 0;
      while (gone > 0 && pre.firstChild) {
        if (pre.firstChild.nodeType === 3) { gone -= 1; }
        pre.removeChild(pre.firstChild);
      }
      pre.insertAdjacentHTML("beforeend", text.slice(lastLog.length - at));
      around(old, fresh, pre, next);
      if (stays && !follow) {
        pre.scrollTop += within(pre, stays) - was;
      }
    } else {
      var top = pre ? pre.scrollTop : 0;
      old.innerHTML = fresh.innerHTML;
      pre = log(old);
      if (pre) { pre.scrollTop = top; }
    }
    if (pre && follow) { pre.scrollTop = pre.scrollHeight; }
    lastLog = text;
  }

  function updatePasses(old, fresh) {
    var open = [];
    each.call(old.querySelectorAll("details"), function (item) {
      open.push(item.open);
    });
    old.innerHTML = fresh.innerHTML;
    each.call(old.querySelectorAll("details"), function (item, index) {
      if (index < open.length) { item.open = open[index]; }
    });
  }

  function update(old, fresh) {
    var name = old.getAttribute("data-part");
    if (fresh.innerHTML === last[name]) { return; }
    if (name === "log") {
      updateLog(old, fresh);
    } else if (name === "passes") {
      updatePasses(old, fresh);
    } else {
      old.innerHTML = fresh.innerHTML;
    }
    last[name] = fresh.innerHTML;
  }

  function again() { window.setTimeout(tick, wait); }

  function tick() {
    fetch(window.location.pathname, { credentials: "same-origin", cache: "no-store" })
      .then(function (response) {
        if (!response.ok) { throw new Error("not read"); }
        return response.text();
      })
      .then(function (text) {
        var page = new DOMParser().parseFromString(text, "text/html");
        var next = page.getElementById("series");
        // A question, the end of the series or another page: load it whole.
        if (!next || !next.hasAttribute("data-live") || names(next) !== names(root)) {
          window.location.reload();
          return;
        }
        var fresh = parts(next);
        each.call(parts(root), function (old, index) { update(old, fresh[index]); });
        document.title = page.title;
        again();
      })
      .catch(again);
  }

  again();
})();
"""
