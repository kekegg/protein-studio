// Section 02 - 整体架构 (Layered architecture)
// Architecture content slide: 5 horizontal layer bars + side notes

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '02  -  整体架构',
    '分层单一职责,自下而上:数据 IO  ->  计算  ->  引擎  ->  UI  ->  Agent 控制面');
  addSectionMarker(slide, theme, '02', 'ARCHITECTURE');

  // Left: 5 stacked layer bars
  const layers = [
    { name: 'Agent Control Layer',  tech: 'MCP stdio  +  HTTP  +  Qt transport',          file: 'src/protein_studio/mcp_server.py (735 L)' },
    { name: 'UI Layer',             tech: 'PySide6 6.11  +  QMainWindow + dock widgets', file: 'src/protein_studio/ui/  (10 modules)' },
    { name: 'Engine Layer',         tech: 'PyVista (VTK)  +  pyvistaqt QtInteractor',     file: 'src/protein_studio/ui/viewport.py (922 L)' },
    { name: 'Compute Layer',        tech: 'OpenMM 8.6  +  AutoDock Vina  +  EquiScore',   file: 'src/protein_studio/core/ (15 modules)' },
    { name: 'IO Layer',             tech: 'PDB / PDBQT / DCD / POS  parsers',            file: 'src/protein_studio/core/io.py + trajectory.py' }
  ];

  const barX = 0.4;
  const barW = 5.7;
  const barH = 0.62;
  const startY = 1.40;
  const gap = 0.10;

  layers.forEach((L, i) => {
    const by = startY + i * (barH + gap);

    // Background panel
    slide.addShape(pres.shapes.RECTANGLE, {
      x: barX, y: by, w: barW, h: barH,
      fill: { color: (i === 0) ? theme.accent : theme.light },
      line: { type: 'none' },
      rectRadius: 0.05
    });

    // Layer name
    slide.addText(L.name, {
      x: barX + 0.20, y: by + 0.05, w: 2.7, h: 0.30,
      fontFace: 'Microsoft YaHei',
      fontSize: 13,
      bold: true,
      color: (i === 0) ? 'FFFFFF' : theme.primary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // Tech stack
    slide.addText(L.tech, {
      x: barX + 0.20, y: by + 0.32, w: 3.6, h: 0.26,
      fontFace: 'Consolas',
      fontSize: 10,
      color: (i === 0) ? 'C5D2E8' : theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });

    // File reference (right-aligned, monospace)
    slide.addText(L.file, {
      x: barX + 3.80, y: by + 0.18, w: barW - 4.00, h: 0.30,
      fontFace: 'Consolas',
      fontSize: 9,
      color: (i === 0) ? 'C5D2E8' : theme.accent,
      align: 'right',
      valign: 'middle',
      margin: 0
    });
  });

  // Right side: design principles
  const rx = 6.40, rw = 3.20;
  slide.addText('设计原则', {
    x: rx, y: 1.40, w: rw, h: 0.35,
    fontFace: 'Microsoft YaHei',
    fontSize: 14,
    bold: true,
    color: theme.primary,
    margin: 0
  });

  const principles = [
    { t: '同进程', d: 'UI 与计算共用 Python 进程,避免跨进程 context 传递' },
    { t: 'API-first', d: 'API 层是 source of truth,GUI 只是 API 的一个调用方' },
    { t: '三层防护', d: '计算作业加入 retry/drop/minimise 三层容错' },
    { t: '单一职责', d: '每个核心模块只负责一种表达或一种 IO 格式' }
  ];
  principles.forEach((p, i) => {
    const py = 1.85 + i * 0.75;
    slide.addText(
      [
        { text: p.t + '  ', options: { fontFace: 'Consolas', color: theme.accent, bold: true, fontSize: 11 } },
        { text: p.d, options: { fontFace: 'Microsoft YaHei', color: theme.secondary, fontSize: 10 } }
      ],
      {
        x: rx, y: py, w: rw, h: 0.70,
        align: 'left',
        valign: 'top',
        margin: 0
      }
    );
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 4, 10);
}

module.exports = { createSlide };
