// TOC - sidebar navigation style with numbered sections
// 5 sections; vertical list on left, project stats panel on right

const { addPageNumber, addFooter, addSlideTitle } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '目录 Contents', '5 sections  -  ~10 minutes  -  focus on architecture & engineering decisions');

  // Left side - numbered sections
  const sections = [
    { num: '01', title: '项目定位 & 现状', desc: 'Why this project, current scale', page: 'p.03' },
    { num: '02', title: '整体架构', desc: 'UI / Engine / Compute / IO / Control layers', page: 'p.04' },
    { num: '03', title: '关键模块 - 表达矩阵', desc: '7 reps x 7 schemes, SS-aware ribbon', page: 'p.05' },
    { num: '04', title: 'FEP 计算引擎', desc: 'NaN 三层防护 + MD sweep pipeline', page: 'p.06' },
    { num: '05', title: 'MCP 控制平面', desc: '让 AI Agent 直接驱动工具链', page: 'p.07' }
  ];

  const startY = 1.50;
  const rowH = 0.62;
  sections.forEach((s, i) => {
    const y = startY + i * rowH;

    // Big number on the left
    slide.addText(s.num, {
      x: 0.4, y: y, w: 0.9, h: rowH - 0.10,
      fontFace: 'Georgia',
      fontSize: 30,
      bold: true,
      color: theme.accent,
      align: 'left',
      valign: 'middle',
      margin: 0
    });

    // Thin vertical separator
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 1.35, y: y + 0.08, w: 0.025, h: rowH - 0.24,
      fill: { color: theme.light },
      line: { type: 'none' }
    });

    // Title
    slide.addText(s.title, {
      x: 1.55, y: y, w: 4.6, h: 0.32,
      fontFace: 'Microsoft YaHei',
      fontSize: 16,
      bold: true,
      color: theme.primary,
      align: 'left',
      valign: 'bottom',
      margin: 0
    });
    // Description
    slide.addText(s.desc, {
      x: 1.55, y: y + 0.30, w: 4.6, h: 0.26,
      fontFace: 'Calibri',
      fontSize: 11,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });

    // Page reference on far right
    slide.addText(s.page, {
      x: 6.40, y: y, w: 0.7, h: rowH - 0.10,
      fontFace: 'Consolas',
      fontSize: 11,
      color: theme.secondary,
      align: 'right',
      valign: 'middle',
      margin: 0
    });
  });

  // Right panel - project at-a-glance stats card
  const panelX = 7.30, panelY = 1.50, panelW = 2.30, panelH = 3.40;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: panelX, y: panelY, w: panelW, h: panelH,
    fill: { color: theme.light },
    line: { type: 'none' },
    rectRadius: 0.08
  });

  slide.addText('PROJECT AT A GLANCE', {
    x: panelX + 0.20, y: panelY + 0.20, w: panelW - 0.40, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 10,
    bold: true,
    charSpacing: 3,
    color: theme.accent,
    margin: 0
  });

  const stats = [
    { v: '15,868', l: 'src Python LOC' },
    { v: '119', l: 'tracked files' },
    { v: '161/162', l: 'tests pass' },
    { v: '10', l: 'modified (this round)' }
  ];
  stats.forEach((s, i) => {
    const sy = panelY + 0.65 + i * 0.62;
    slide.addText(s.v, {
      x: panelX + 0.20, y: sy, w: panelW - 0.40, h: 0.32,
      fontFace: 'Georgia',
      fontSize: 22,
      bold: true,
      color: theme.accent,
      align: 'left',
      valign: 'top',
      margin: 0
    });
    slide.addText(s.l, {
      x: panelX + 0.20, y: sy + 0.32, w: panelW - 0.40, h: 0.22,
      fontFace: 'Calibri',
      fontSize: 10,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 2, 10);
}

module.exports = { createSlide };
