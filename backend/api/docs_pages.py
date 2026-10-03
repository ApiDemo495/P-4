"""README and docs rendered as HTML inside the app (Round S).

The user could not open the editor's markdown preview ("command
markdown.showPreview not found" - the VS Code Markdown extension was not
available in the Codespace).  Nothing in the repository can install an editor
extension, so the guide is served by the engine itself at /readme.
"""
from __future__ import annotations

import html
from pathlib import Path

from backend.core import config as cfg

#: Pages that may be served, by short name.  Nothing outside this map is read.
PAGES: dict[str, tuple[str, Path]] = {
    "README": ("README", cfg.REPO_ROOT / "README.md"),
    "keys": ("API keys & models", cfg.REPO_ROOT / "docs" / "API_KEYS_AND_MODELS.md"),
    "spec": ("Specification v2", cfg.REPO_ROOT / "docs" / "SPECIFICATION_v2.md"),
    "notes": ("Spec notes (per round)", cfg.REPO_ROOT / "docs" / "SPEC_NOTES.md"),
}

_CSS = """
body{margin:0;background:#0b0f14;color:#d8dee9;font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}
nav{display:flex;gap:16px;align-items:center;padding:12px 24px;background:#111823;border-bottom:1px solid #1f2a3a;position:sticky;top:0}
nav a{color:#9fb3c8;text-decoration:none;font-size:13px}nav a.active,nav a:hover{color:#fff}
main{max-width:920px;margin:0 auto;padding:28px 24px 80px}
h1,h2,h3{color:#fff;line-height:1.25}h1{font-size:28px}h2{margin-top:36px;border-bottom:1px solid #1f2a3a;padding-bottom:6px}
pre{background:#0f1620;border:1px solid #1f2a3a;border-radius:8px;padding:12px 14px;overflow:auto;position:relative}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#e5e9f0}
p code,li code{background:#182230;padding:1px 5px;border-radius:4px}
table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #1f2a3a;padding:6px 10px;text-align:left}
th{background:#111823}a{color:#7cc4ff}blockquote{border-left:3px solid #2b3a4f;margin:0;padding:2px 14px;color:#9fb3c8}
.copy{position:absolute;top:6px;right:6px;font-size:11px;background:#1f2a3a;color:#d8dee9;border:0;border-radius:4px;padding:3px 8px;cursor:pointer}
.missing{color:#f87171}
"""

_JS = """
document.querySelectorAll('pre').forEach((pre) => {
  const b = document.createElement('button'); b.className = 'copy'; b.textContent = 'copy';
  b.onclick = () => { navigator.clipboard.writeText(pre.innerText.replace(/copy$/, '').trim()); b.textContent = 'copied'; setTimeout(() => (b.textContent = 'copy'), 1200); };
  pre.appendChild(b);
});
"""


def _to_html(text: str) -> str:
    try:
        import markdown  # in requirements.txt; the fallback keeps the page readable without it

        return markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    except ImportError:
        return "<pre><code>" + html.escape(text) + "</code></pre>"


def render(name: str) -> tuple[str, int]:
    """(html, status) for page ``name``; unknown or missing pages are 404."""
    entry = PAGES.get(name) or PAGES.get(name.upper())
    status = 200
    if entry is None:
        title, body = "Not found", f'<p class="missing">No page called {html.escape(name)}.</p>'
        status = 404
    else:
        title, path = entry
        try:
            body = _to_html(path.read_text(encoding="utf-8"))
        except OSError:
            body, status = f'<p class="missing">{html.escape(str(path.name))} is missing from this checkout.</p>', 404
    links = "".join(
        f'<a href="/readme/{key}" class="{"active" if key == name or (name.upper() == key) else ""}">{html.escape(label)}</a>'
        for key, (label, _p) in PAGES.items()
    )
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)} · DROSOPHILA TRADER</title>"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<style>{_CSS}</style></head><body>"
        f'<nav><a href="/">← Dashboard</a>{links}<a href="/docs" target="_blank">API</a></nav>'
        f"<main>{body}</main><script>{_JS}</script></body></html>"
    )
    return page, status
