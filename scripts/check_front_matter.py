"""Word count and style check on the abstract and highlights.

Elsevier caps the abstract at 250 words and highlights at 85 characters each.
Also flags semicolon-joined sentences and em dashes, which are house style here.
"""

import re
from pathlib import Path

BS = chr(92)
p = Path(__file__).resolve().parent.parent / "mri_revision.tex"
t = p.read_text(encoding="utf-8")

a = t[t.index(BS + "begin{abstract}") + 16: t.index(BS + "end{abstract}")]
plain = re.sub(r"\$[^$]*\$", "X", a)
plain = re.sub(BS + r"[a-zA-Z]+", "", plain)
n = len(plain.split())
print(f"abstract: {n} words {'OK' if n <= 250 else 'OVER LIMIT'}, "
      f"{a.count(';')} semicolons, {a.count('---')} em dashes")

h = t[t.index(BS + "begin{highlights}"): t.index(BS + "end{highlights}")]
items = re.findall(BS + BS + r"item (.+)", h)
print(f"highlights: {len(items)}")
for ln in items:
    c = ln.replace(BS, "")
    print(f"  {'OK  ' if len(c) <= 85 else 'LONG'} {len(c):>3d}  {c}")

# BS + "title" builds the regex \\title, where \\t is a tab escape, so the
# pattern could never match. Double the backslash to mean a literal one.
print(f"\ntitle: {re.search(BS + BS + r'title\[mode=title\]\{(.+?)\}$', t, re.M).group(1)}")
