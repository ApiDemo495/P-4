"""Layout guard: a state class must never take a widget out of the page.

Round-3 regression.  The dashboard put the class ``emergency`` on the HOLD box
while an override was active, and ``styles.css`` still carried a leftover rule

    .emergency { position: fixed; inset: 0; z-index: 100; ... }

from the days when the override was a full-screen card.  The rule matched the
widget, moved it out of its grid cell and stretched it over the viewport: the
user saw a red, glittering screen and no app.  Deleting the overlay markup was
not enough, because the *selector* stayed behind.

These tests fail the build if that pattern ever comes back:

1.  every rule that declares ``position: fixed`` (or a fixed ``inset``) may only
    be matched by dialog markup - currently exactly ``.modal``;
2.  no class that the client scripts apply may be a bare selector that goes
    full-screen;
3.  the HOLD box keeps an inline, height-capped box style.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
PAGES = ("index.html", "matrix.html", "settings.html")
SCRIPTS = ("app.js", "matrix.js", "settings.js")

# Full-screen layers that are legitimate: dialogs, opened deliberately, closed
# by a button.  Keyed by the compound selector that may carry the rule.
ALLOWED_FULLSCREEN_SELECTORS = {".modal"}


# --------------------------------------------------------------------------
# tiny, tolerant CSS reader (no nesting in this stylesheet except @keyframes)
# --------------------------------------------------------------------------
def _strip_at_blocks(css: str) -> str:
    out, i, depth = [], 0, 0
    while i < len(css):
        if css.startswith("@", i) and depth == 0:
            at = css.find("{", i)
            if at == -1:
                break
            name = css[i:at].strip()
            if name.startswith("@keyframes") or name.startswith("@media"):
                j, level = at, 0
                while j < len(css):
                    if css[j] == "{":
                        level += 1
                    elif css[j] == "}":
                        level -= 1
                        if level == 0:
                            break
                    j += 1
                if name.startswith("@media"):
                    # @media bodies still contain real rules - keep them.
                    out.append(_strip_at_blocks(css[at + 1 : j]))
                i = j + 1
                continue
        out.append(css[i])
        i += 1
    return "".join(out)


def css_rules() -> list[tuple[str, str]]:
    text = _strip_at_blocks((WEB / "styles.css").read_text())
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    rules = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text):
        selector = " ".join(selector.split())
        if selector:
            rules.append((selector, body))
    assert rules, "styles.css produced no rules - the parser is broken"
    return rules


def class_tokens_applied_by_scripts() -> set[str]:
    tokens: set[str] = set()
    for name in SCRIPTS:
        js = (WEB / name).read_text()
        for args in re.findall(r"classList\.(?:add|remove|toggle)\(([^)]*)\)", js):
            tokens.update(re.findall(r"""["'`]([\w-]+)["'`]""", args))
        for line in js.splitlines():
            if "className" in line and "=" in line:
                tokens.update(re.findall(r"""["'`]([\w-]+)["'`]""", line))
    return tokens


def class_tokens_in_pages() -> set[str]:
    tokens: set[str] = set()
    for page in PAGES:
        html = (WEB / page).read_text()
        for attr in re.findall(r'class="([^"]*)"', html):
            tokens.update(attr.split())
    return tokens


def _classes_in(compound: str) -> set[str]:
    return set(re.findall(r"\.([A-Za-z_][\w-]*)", compound))


def _html_classes(compound: str) -> set[str]:
    """Class tokens of the element part of a selector (ignore descendants)."""
    last = compound.split()[-1] if compound.split() else compound
    return _classes_in(last)


def test_no_page_class_goes_fullscreen() -> None:
    """A rule that pins an element to the viewport must not match page content."""
    dangerous = class_tokens_applied_by_scripts() | class_tokens_in_pages()
    offenders = []
    for selector, body in css_rules():
        pinned = re.search(r"position\s*:\s*fixed", body)
        if not pinned:
            continue
        for compound in selector.split(","):
            compound = compound.strip()
            if compound in ALLOWED_FULLSCREEN_SELECTORS:
                continue
            hit = _html_classes(compound) & dangerous
            if hit:
                offenders.append(f"{compound} {{ ...{hit}... }} (full-screen rule)")
    assert not offenders, (
        "these rules can lift a normal page element over the whole screen - "
        "scope them to their element or delete them: " + "; ".join(offenders)
    )


def test_no_bare_class_sets_position_fixed() -> None:
    """`.something { position: fixed }` is the shape of the original bug."""
    for selector, body in css_rules():
        if not re.search(r"position\s*:\s*fixed", body):
            continue
        for compound in selector.split(","):
            compound = compound.strip()
            if compound == ".modal" or compound.endswith(" .modal"):
                continue
            assert not re.fullmatch(r"\.[\w-]+", compound), (
                f"bare class rule {compound} goes full-screen; a client script "
                f"that adds this class will cover the page"
            )


def test_emergency_override_class_is_scoped() -> None:
    """The override state is `.hold-box.override`, never a global name."""
    applied = class_tokens_applied_by_scripts()
    assert "emergency" not in applied, (
        "JS applies a bare `emergency` class again - that name used to match a "
        "leftover full-screen rule in styles.css"
    )
    assert "override" in applied, "the override state class disappeared"

    bare = [
        selector
        for selector, _ in css_rules()
        if any(re.fullmatch(r"\.emergency|\.emergency[\s,{]", c.strip()) for c in selector.split(","))
    ]
    assert not bare, f"global .emergency rules exist again: {bare}"


def test_hold_box_is_inline_and_bounded() -> None:
    rules = {sel: body for sel, body in css_rules()}
    base = rules.get(".hold-box")
    assert base, ".hold-box rule is missing"
    assert "position: static" in base, (
        "the HOLD box must stay in the document flow (position: static)"
    )
    assert "max-height" in base, (
        "the HOLD box needs a max-height so a long headline scrolls instead of "
        "growing over the panel"
    )
    override = rules.get(".hold-box.override")
    assert override, ".hold-box.override rule is missing"
    assert "position" not in override and "inset" not in override, (
        "the override style must only recolour the box - no positioning"
    )


def test_pages_stamp_their_assets() -> None:
    """A cached stylesheet must not be able to hide a layout fix."""
    for page in PAGES:
        html = (WEB / page).read_text()
        for asset in ("/static/styles.css", "/static/app.js", "/static/matrix.js", "/static/settings.js"):
            if asset not in html:
                continue
            assert re.search(re.escape(asset) + r"\?v=", html), (
                f"{page} loads {asset} without a version stamp"
            )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
