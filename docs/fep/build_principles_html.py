#!/usr/bin/env python3
"""Build docs/fep/FEP_principles.html from FEP_principles.docx.

Renders display formulas via hand-crafted LaTeX in \\[...\\].
Prose paragraphs get light inline-math wrapping: any token containing a
Greek letter, math operator, or LaTeX command is merged with adjacent
math tokens into a single \\(...\\) region. Plain prose stays plain.

Run from the protein-studio root:
    python docs/fep/build_principles_html.py
"""

import re
import html as html_lib
from pathlib import Path
from docx import Document

HERE = Path(__file__).resolve().parent
DOCX_PATH = HERE / 'FEP_principles.docx'
HTML_PATH = HERE / 'FEP_principles.html'

# === Display formulas (paragraph index -> LaTeX) ============================
DISPLAY_FORMULAS = {
    13: r"H(\lambda) \;=\; (1-\lambda)\, H_A \;+\; \lambda\, H_B",
    15: r"\Delta G(A\!\to\!B) \;=\; \int_0^1 \!\left\langle \tfrac{\partial H}{\partial \lambda}\!\right\rangle_{\!\lambda}\, d\lambda",
    17: r"\Delta G \;=\; k_B T \,\cdot\, \frac{\ln\big\langle e^{-\beta\,\Delta U}\big\rangle_A}{\big\langle e^{+\beta\,\Delta U}\big\rangle_B}",
    27: r"U(r;\lambda) \;=\; \lambda\!\cdot\!4\varepsilon\!\left[\,\frac{\sigma_{12}^{12}}{\big(\alpha(1-\lambda)^2 + r^6\big)^2} \;-\; \frac{\sigma_{12}^{6}}{\alpha(1-\lambda)^2 + r^6}\,\right]",
    31: r"q_{\mathrm{eff}}(\lambda) \;=\; q \cdot \big(1 - \mathrm{is\_dis}\!\cdot\!\lambda\big)",
    35: r"\sum_A \frac{1}{1+\exp\!\big(\beta(\Delta U + C)\big)} \;=\; \sum_B \frac{1}{1+\exp\!\big(-\beta(\Delta U + C)\big)}",
    39: r"\Delta G_{\mathrm{corrected}} \;=\; \Delta G_{\mathrm{FEP}} \;+\; \Delta G_{\mathrm{restraint}}",
    41: r"\Delta G_{\mathrm{restraint}} \;=\; N \cdot k_B T \,\cdot\, \left[\,\tfrac{3}{2}\ln\!\left(\tfrac{2\pi k_B T}{k}\right) - \ln(V_0)\,\right]",
    44: r"\Delta G_{\mathrm{bind}} \;=\; \Delta G_{\mathrm{complex}} - \Delta G_{\mathrm{solution}} + \Delta G_{\mathrm{Boresch}} + \Delta G_{\mathrm{std\,state}}",
}

# === Unicode symbol -> LaTeX ===============================================
SYMBOL_MAP = {
    'Δ': r'\Delta ',  'λ': r'\lambda ',  'ε': r'\varepsilon ', 'σ': r'\sigma ',
    'α': r'\alpha ',  'β': r'\beta ',   'π': r'\pi ',         'Σ': r'\sum ',
    '∫': r'\int ',    '∂': r'\partial ', '·': r'\cdot ',      '×': r'\times ',
    '→': r'\to ',     '∈': r'\in ',     '⟨': r'\langle ',    '⟩': r'\rangle ',
    '≠': r'\ne ',     '≤': r'\le ',     '≥': r'\ge ',        '−': '-',
    '₁₂': '_{12}',    '₀': '_0',        '₁': '_1',           '₂': '_2',
    '⁰': '^0',        '¹': '^1',        '²': '^2',           '⁶': '^6',
    '¹²': '^{12}',
}

MATH_CHARS = set('ΔλΣ∫∂·×∈≤≥≠παβεσ\\^_')

# === Math conversion ========================================================
def to_latex(text: str) -> str:
    """Convert plain-text math notation to inline LaTeX."""
    s = text
    # Phase 1: Unicode -> LaTeX commands
    for k, v in SYMBOL_MAP.items():
        s = s.replace(k, v)

    # Phase 2: single math variable + exactly one underscore pair (math subscript).
    # Requires word boundary, optional whitespace between variable and ``_``,
    # and that the subscript is NOT followed by ``_`` / digit / letter.
    s = re.sub(
        r'(?<![A-Za-z\\])(\\[A-Za-z]+|[A-Za-z])(?![a-zA-Z])\s*_([A-Za-z][A-Za-z0-9]*)(?![A-Za-z0-9_])',
        lambda m: m.group(1) + '_{\\mathrm{' + m.group(2) + '}}',
        s,
    )

    # Phase 3: 2+ underscore pairs (programmatic identifier).
    # n_equilibration_steps, sample_neighbor_lambdas, etc.
    s = re.sub(
        r'([A-Za-z\\][A-Za-z]*)(_[a-zA-Z][A-Za-z0-9]*){2,}',
        lambda m: '\\mathrm' + '{' + m.group(0).replace('_', '\\_') + '}',
        s,
    )

    # Phase 4: multi-letter prefix + single underscore (programmatic name).
    # is_dis, boresch_correction, etc.
    s = re.sub(
        r'([A-Za-z][A-Za-z]+)_([a-zA-Z][A-Za-z0-9]*)',
        lambda m: '\\mathrm' + '{' + m.group(0).replace('_', '\\_') + '}',
        s,
    )

    # Phase 5: single uppercase subscripts: H_A, q_A
    s = re.sub(
        r'\b([A-Za-z\\])_([A-Z])\b',
        lambda m: m.group(1) + '_{' + m.group(2) + '}',
        s,
    )

    # Phase 6: Numeric subscripts after letter or LaTeX command.
    # r_0, lambda_0, G_12 -> G_{12}
    s = re.sub(
        r'(?<=[\w\\])_(\d+)',
        lambda m: '_{' + m.group(1) + '}',
        s,
    )

    # Phase 7: Numeric superscripts after letter/LaTeX command.
    s = re.sub(
        r'([A-Za-z0-9\\])\^(\d+)',
        lambda m: m.group(1) + '^{' + m.group(2) + '}',
        s,
    )

    # Phase 8: Bare superscripts (from unicode ¹² ⁶ etc.).
    s = re.sub(
        r'(?<!\\)\^(\d+)',
        lambda m: '^{' + m.group(1) + '}',
        s,
    )

    # Phase 9: Escape remaining underscores (programmatic names like is_dis),
    # but preserve LaTeX subscript syntax ``_{...}``.
    s = re.sub(r'(?<!\\)_(?!\{)', r'\\_', s)

    # Normalize whitespace (collapses double spaces introduced by
    # trailing-space SYMBOL_MAP substitutions).
    s = re.sub(r'[ \t]{2,}', ' ', s)
    s = re.sub(r' *\n *', '\n', s)

    return s


def token_is_math(tok: str) -> bool:
    """A token is math if it contains any math-related char."""
    return any(c in MATH_CHARS for c in tok)


def wrap_inline_math(s: str) -> str:
    """Merge adjacent math tokens into a single \\(...\\) region.

    Spreads math to single uppercase letter variables and common math
    function-call patterns so that equations like ``U(state B) = ...`` come
    out as a single math span rather than several broken pieces.
    """
    tokens = re.split(r'(\s+)', s)
    out = []
    in_math = False
    math_buf = []

    def flush():
        nonlocal in_math
        if math_buf:
            content = ''.join(math_buf)
            stripped = content.rstrip()
            trailing_ws = content[len(stripped):]
            if stripped:
                out.append('\\(' + stripped + '\\)')
            if trailing_ws:
                out.append(trailing_ws)
            math_buf.clear()
        in_math = False

    def adjacent_nonws(idx, direction):
        j = idx + direction
        while 0 <= j < len(tokens) and tokens[j].isspace():
            j += direction
        return tokens[j] if 0 <= j < len(tokens) else ''

    for i, tok in enumerate(tokens):
        if not tok:
            continue
        if tok.isspace():
            if in_math:
                math_buf.append(tok)
            else:
                out.append(tok)
            continue

        is_math = token_is_math(tok)

        # Single uppercase letter: math if adjacent to a math token.
        if not is_math and re.fullmatch(r'[A-Z]', tok):
            prev = adjacent_nonws(i - 1, -1)
            next_ = adjacent_nonws(i + 1, +1)
            if token_is_math(prev) or token_is_math(next_):
                is_math = True

        # Function-call pattern in math: U(state, G(x)
        if not is_math and re.match(r'^[A-Z][A-Za-z]*\(', tok):
            is_math = True

        # Single letter + closer inside a math region: keep inside.
        if not is_math and in_math and re.fullmatch(r'[A-Z][)\].,;]+', tok):
            is_math = True

        if is_math:
            if not in_math:
                in_math = True
            math_buf.append(tok)
        elif in_math:
            # Trailing closer punctuation stays in math.
            if tok in (')', ']', '}', '.', ','):
                math_buf.append(tok)
            else:
                flush()
                out.append(tok)
        else:
            out.append(tok)
    flush()
    return ''.join(out)


def render_inline(text: str) -> str:
    """Render prose text with light inline math wrapping."""
    s = to_latex(text)
    s = wrap_inline_math(s)
    # HTML-escape but preserve \(, \), \_, \\, \Delta, \lambda, etc.
    return html_lib.escape(s, quote=False)


# === Body builder ===========================================================
def build_body(paragraphs):
    out = []
    in_ul = False
    title_seen = False
    subtitle_seen = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append('</ul>')
            in_ul = False

    for i, p in enumerate(paragraphs):
        style = p.style.name
        text = p.text.strip()

        if not text and style != 'List Bullet':
            close_ul()
            continue

        # The docx title and subtitle are rendered in the page header
        # already — skip them here to avoid duplication.
        if style == 'Title':
            close_ul()
            title_seen = True
            continue
        if (
            style == 'Normal'
            and title_seen
            and not subtitle_seen
            and i == 1
        ):
            close_ul()
            subtitle_seen = True
            continue
        if style == 'Heading 1':
            close_ul()
            anchor = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
            out.append(
                f'<h2 class="part" id="{anchor}">{html_lib.escape(text)}</h2>'
            )
        elif style == 'Heading 2':
            close_ul()
            anchor = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
            out.append(
                f'<h3 class="section" id="{anchor}">{html_lib.escape(text)}</h3>'
            )
        elif style == 'Heading 3':
            close_ul()
            anchor = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
            out.append(
                f'<h4 class="subsection" id="{anchor}">{html_lib.escape(text)}</h4>'
            )
        elif style == 'List Bullet':
            if not in_ul:
                out.append('<ul>')
                in_ul = True
            out.append(f'<li>{render_inline(text)}</li>')
        elif style == 'Normal':
            close_ul()
            if i in DISPLAY_FORMULAS:
                out.append(
                    f'<p class="formula">\\[{DISPLAY_FORMULAS[i]}\\]</p>'
                )
            else:
                out.append(f'<p>{render_inline(text)}</p>')

    close_ul()
    return '\n'.join(out)


# === TOC builder ============================================================
def build_toc(paragraphs):
    items = []
    for p in paragraphs:
        style = p.style.name
        text = p.text.strip()
        if not text or style not in ('Heading 1', 'Heading 2', 'Heading 3'):
            continue
        anchor = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
        css = {'Heading 1': 'l1', 'Heading 2': 'l2', 'Heading 3': 'l3'}[style]
        items.append(
            f'<li class="{css}"><a href="#{anchor}">{html_lib.escape(text)}</a></li>'
        )
    return '<ul class="toc-list">\n' + '\n'.join(items) + '\n</ul>'


# === HTML template ==========================================================
HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FEP — Free Energy Perturbation · protein-studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true,
    tags: 'ams',
    macros: {
      ss: '\\mathrm{ss}',
      FEP: '\\mathrm{FEP}'
    }
  },
  svg: { fontCache: 'global' },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        document.body.classList.add('mathjax-ready');
      });
    }
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root {
  --fg: #1d1f23;
  --fg-soft: #5b6068;
  --fg-mute: #8b9099;
  --accent: #2257b1;
  --accent-soft: #e9effa;
  --bg: #fdfdfd;
  --formula-bg: #f6f8fc;
  --border: #d8dce3;
  --code-bg: #f3f4f7;
  --link: #2257b1;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
               Roboto, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.72;
  font-size: 16px;
  -webkit-text-size-adjust: 100%;
}

header.page-header {
  border-bottom: 1px solid var(--border);
  padding: 2.2rem 2rem 1.4rem;
  background: linear-gradient(180deg, #f6f8fb 0%, #fdfdfd 100%);
}
header.page-header h1 {
  font-size: 2.1rem;
  margin: 0 0 0.4rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: #1d1f23;
}
header.page-header .subtitle {
  font-size: 1.02rem;
  color: var(--fg-soft);
  margin: 0;
}

nav.toc {
  background: var(--accent-soft);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 1.6rem 2rem 2rem;
  padding: 1.1rem 1.5rem;
}
nav.toc h2 {
  margin: 0 0 0.7rem;
  font-size: 0.84rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
}
ul.toc-list {
  list-style: none;
  padding: 0;
  margin: 0;
  columns: 2;
  column-gap: 2rem;
}
ul.toc-list li.l1 {
  font-weight: 600;
  margin-top: 0.55rem;
  break-inside: avoid;
  color: #1d1f23;
}
ul.toc-list li.l2,
ul.toc-list li.l3 {
  font-size: 0.94em;
  padding-left: 1.1rem;
  color: var(--fg-soft);
}
ul.toc-list li.l3 { padding-left: 2rem; }
ul.toc-list a {
  color: inherit;
  text-decoration: none;
  display: block;
  padding: 0.12rem 0;
}
ul.toc-list a:hover {
  color: var(--accent);
  text-decoration: underline;
}

main {
  max-width: 820px;
  margin: 0 auto;
  padding: 0 1.4rem 4rem;
}

h2.part {
  font-size: 1.55rem;
  margin: 3.2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent);
  color: var(--accent);
  font-weight: 700;
  letter-spacing: -0.005em;
}

h3.section {
  font-size: 1.22rem;
  margin: 2.2rem 0 0.7rem;
  color: var(--fg);
  font-weight: 600;
}

h4.subsection {
  font-size: 1.04rem;
  margin: 1.55rem 0 0.5rem;
  color: var(--fg-soft);
  font-weight: 600;
}

p { margin: 0.7rem 0; }
.doc-subtitle {
  color: var(--fg-soft);
  font-size: 1.05rem;
  margin: 0.3rem 0 0;
  font-style: italic;
}

ul { padding-left: 1.6rem; margin: 0.5rem 0; }
li { margin: 0.3rem 0; }

.formula {
  background: var(--formula-bg);
  border-left: 3px solid var(--accent);
  padding: 0.85rem 1.2rem;
  margin: 1.25rem 0;
  border-radius: 0 6px 6px 0;
  overflow-x: auto;
  text-align: center;
}
.formula mjx-container {
  margin: 0 !important;
  display: inline-block;
}

code {
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: 0.9em;
  background: var(--code-bg);
  padding: 0.06rem 0.34rem;
  border-radius: 3px;
  color: #4a3d2a;
}

footer.page-footer {
  border-top: 1px solid var(--border);
  padding: 1.5rem 2rem;
  margin-top: 4rem;
  color: var(--fg-mute);
  font-size: 0.86rem;
  text-align: center;
  background: #fafafa;
}

a { color: var(--link); }

/* anchor offsets so headings aren't hidden behind any sticky element */
h2.part, h3.section, h4.subsection {
  scroll-margin-top: 1rem;
}

@media (max-width: 760px) {
  ul.toc-list { columns: 1; }
  main { padding: 0 1rem 3rem; }
  header.page-header { padding: 1.5rem 1rem; }
  nav.toc { margin: 1rem 1rem 1.5rem; }
  .formula { padding: 0.7rem 0.9rem; }
}

@media print {
  body { font-size: 11pt; line-height: 1.5; }
  header.page-header, nav.toc, footer.page-footer { display: none; }
  main { max-width: none; padding: 0; }
  .formula { border: 1px solid #ccc; background: #f9f9f9; page-break-inside: avoid; }
  h2.part, h3.section, h4.subsection { break-after: avoid; }
  a { color: inherit; text-decoration: none; }
}
</style>
</head>
<body>

<header class="page-header">
  <h1>FEP — Free Energy Perturbation</h1>
  <p class="subtitle">Alchemical Free Energy Calculations in protein-studio</p>
</header>

<nav class="toc">
  <h2>Contents</h2>
  ___TOC___
</nav>

<main>
___BODY___
</main>

<footer class="page-footer">
  Generated from <code>FEP_principles.docx</code> · protein-studio FEP framework ·
  rendered with MathJax 3
</footer>

</body>
</html>
"""


def main():
    doc = Document(str(DOCX_PATH))
    paragraphs = list(doc.paragraphs)
    body = build_body(paragraphs)
    toc = build_toc(paragraphs)
    full = HTML_HEAD.replace('___TOC___', toc).replace('___BODY___', body)
    HTML_PATH.write_text(full, encoding='utf-8')
    print(f'Wrote {HTML_PATH} ({len(full):,} bytes)')


if __name__ == '__main__':
    main()