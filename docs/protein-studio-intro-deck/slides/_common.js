// Shared helpers - synchronous, no shared mutable option objects

function addPageNumber(slide, theme, current, total) {
  // Bottom-right page number badge
  slide.addText(
    [
      { text: String(current).padStart(2, '0'), options: { color: theme.accent, bold: true } },
      { text: ' / ' + String(total).padStart(2, '0'), options: { color: theme.secondary } }
    ],
    {
      x: 9.0, y: 5.18, w: 0.8, h: 0.3,
      fontFace: 'Consolas',
      fontSize: 10,
      align: 'right',
      valign: 'middle',
      margin: 0
    }
  );
}

function addFooter(slide, theme) {
  // Bottom-left footer: project name + event label
  slide.addText(
    [
      { text: 'Protein-Studio', options: { color: theme.accent, bold: true } },
      { text: '  -  Internal Tech Review', options: { color: theme.secondary } }
    ],
    {
      x: 0.4, y: 5.18, w: 6.0, h: 0.3,
      fontFace: 'Calibri',
      fontSize: 10,
      align: 'left',
      valign: 'middle',
      margin: 0
    }
  );
}

function addSlideTitle(slide, theme, title, subtitle) {
  // Standard top-of-slide title block (no underline accent - uses whitespace instead)
  slide.addText(title, {
    x: 0.4, y: 0.30, w: 9.2, h: 0.55,
    fontFace: 'Microsoft YaHei',
    fontSize: 24,
    bold: true,
    color: theme.primary,
    align: 'left',
    valign: 'middle',
    margin: 0
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.4, y: 0.85, w: 9.2, h: 0.30,
      fontFace: 'Microsoft YaHei',
      fontSize: 12,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  }
}

function addSectionMarker(slide, theme, sectionNum, sectionLabel) {
  // Small section tag in top-right corner
  slide.addText(
    [
      { text: sectionNum, options: { color: theme.accent, bold: true } },
      { text: '  ' + sectionLabel, options: { color: theme.secondary } }
    ],
    {
      x: 6.4, y: 0.30, w: 3.2, h: 0.40,
      fontFace: 'Consolas',
      fontSize: 11,
      align: 'right',
      valign: 'middle',
      margin: 0
    }
  );
}

module.exports = { addPageNumber, addFooter, addSlideTitle, addSectionMarker };
