"""Brace/paren/bracket balance for Dart sources, with strings and comments skipped."""
import pathlib, sys

def scan(text):
    i, n = 0, len(text)
    depth = {"{": 0, "(": 0, "[": 0}
    pairs = {"}": "{", ")": "(", "]": "["}
    line = 1
    stack = []
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1; i += 1; continue
        if c == "/" and i + 1 < n and text[i+1] == "/":
            while i < n and text[i] != "\n": i += 1
            continue
        if c == "/" and i + 1 < n and text[i+1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i+1] == "/"):
                if text[i] == "\n": line += 1
                i += 1
            i += 2
            continue
        if c in "'\"":
            triple = text[i:i+3] in ("'''", '"""')
            quote = text[i:i+3] if triple else c
            i += len(quote)
            while i < n:
                if text[i] == "\\":
                    i += 2; continue
                if text[i] == "\n":
                    line += 1
                if text[i:i+len(quote)] == quote:
                    i += len(quote); break
                if text[i] == "$" and i + 1 < n and text[i+1] == "{":
                    # interpolation: scan the braces as code
                    depth["{"] += 1; stack.append((line, "{")); i += 2
                    inner = 1
                    while i < n and inner:
                        if text[i] == "{": inner += 1
                        elif text[i] == "}": inner -= 1
                        elif text[i] == "\n": line += 1
                        i += 1
                    depth["{"] -= 1
                    if stack: stack.pop()
                    continue
                i += 1
            continue
        if c in depth:
            depth[c] += 1; stack.append((line, c)); i += 1; continue
        if c in pairs:
            depth[pairs[c]] -= 1
            if stack and stack[-1][1] == pairs[c]:
                stack.pop()
            elif stack:
                return f"line {line}: '{c}' closes '{stack[-1][1]}' opened on line {stack[-1][0]}"
            else:
                return f"line {line}: unmatched '{c}'"
            i += 1; continue
        i += 1
    if stack:
        return f"unclosed {[s[1] for s in stack]} opened at lines {[s[0] for s in stack]}"
    return None

bad = False
for path in sys.argv[1:]:
    err = scan(pathlib.Path(path).read_text())
    print(("OK   " if err is None else "BAD  ") + path + ("" if err is None else "  -> " + err))
    bad = bad or err is not None
sys.exit(1 if bad else 0)
