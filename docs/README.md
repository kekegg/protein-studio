# Documentation generation

This directory ships the developer + user guides as `.docx` and `.pptx`
files, regenerated on demand from a single JavaScript source so both
formats stay in sync.

## Layout

```
docs/
├── generate_docs.mjs     # content + renderers (single source of truth)
├── package.json          # npm dependencies
├── package-lock.json
├── README.md             # this file
├── DeveloperGuide.docx   # generated
├── DeveloperGuide.pptx   # generated
├── UserGuide.docx        # generated
├── UserGuide.pptx        # generated
└── screenshots/          # images referenced by `image` blocks
```

## Toolchain

- [docx](https://www.npmjs.com/package/docx) v9 — programmatic .docx
- [pptxgenjs](https://www.npmjs.com/package/pptxgenjs) v3 — programmatic .pptx
- Node.js 18+

## Install & build

```bash
cd docs
npm install         # one-time
npm run build       # writes the four .docx/.pptx files
# or directly:
node generate_docs.mjs
```

No Python is involved — `generate_docs.py` is kept only as a legacy
reference; the new `generate_docs.mjs` is authoritative.

## Content model

`generate_docs.mjs` exports two data structures (`DEVELOPER_GUIDE` and
`USER_GUIDE`) shaped as a list of `[heading, blocks[]]` pairs. Each
block is one of:

| Kind     | Fields                            | Rendered as                       |
| -------- | --------------------------------- | --------------------------------- |
| `text`   | `text` (string)                   | Body paragraph                    |
| `bullets`| `items` (string[])                | Bulleted list                     |
| `code`   | `code` (string)                   | Monospace block (Consolas, navy box border, green text) |
| `image`  | `file` (string), `widthIn` (num)  | Centered picture; optional extra slide in pptx |

Both renderers share the same palette: navy headings `#103560`, dark
body `#202020`, muted gray captions `#707070`, dark green code `#104010`,
Calibri body, Consolas code.

## Screenshots

Drop new PNGs into `docs/screenshots/` and reference them by file name
in the content blocks. Missing files are silently skipped (the docx/pptx
still build) so a partial screenshot directory is fine during iteration.

## Verifying the outputs

After `npm run build`:

```bash
unzip -p DeveloperGuide.docx word/document.xml | grep -oE 'Overview|MCP server'
unzip -p DeveloperGuide.pptx 'ppt/slides/slide2.xml' | grep -oE 'Overview'
```

If the docx zip won't open in Word/LibreOffice, the most common cause is
an invalid `font` or `color` value in `generate_docs.mjs` — re-check the
constants at the top of the file.