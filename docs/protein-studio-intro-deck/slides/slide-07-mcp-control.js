// Section 05 - MCP 控制平面
// Content - mixed media: 3 transport cards + tools list panel

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '05  -  MCP 控制平面: 让 AI Agent 直接驱动',
    'stdio / HTTP / Qt 三种 transport,所有 GUI / 计算动作通过 MCP tools 暴露 (mcp_server.py 735 L)');
  addSectionMarker(slide, theme, '05', 'MCP CONTROL');

  // Three transport cards on left side
  const transports = [
    { name: 'stdio',  tag: 'default', desc: '与 Claude Desktop 同进程,原生命令通道', port: 'n/a' },
    { name: 'HTTP',   tag: 'multi-client', desc: '多 agent 并发访问,远程部署友好', port: ':7820/mcp' },
    { name: 'Qt',     tag: 'embedded', desc: '同 Python 进程,无 socket 开销', port: 'in-process' }
  ];

  const cardX = 0.40;
  const cardW = 5.20;
  const cardY = 1.45;
  const cardH = 1.15;
  const cardGap = 0.12;

  transports.forEach((t, i) => {
    const y = cardY + i * (cardH + cardGap);
    // Card bg
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: y, w: cardW, h: cardH,
      fill: { color: theme.light },
      line: { type: 'none' },
      rectRadius: 0.05
    });
    // Left accent strip
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: y, w: 0.10, h: cardH,
      fill: { color: theme.accent },
      line: { type: 'none' }
    });
    // Name
    slide.addText(t.name, {
      x: cardX + 0.25, y: y + 0.10, w: 1.30, h: 0.40,
      fontFace: 'Georgia',
      fontSize: 22,
      bold: true,
      color: theme.accent,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // Tag
    slide.addText('// ' + t.tag, {
      x: cardX + 1.50, y: y + 0.10, w: 2.0, h: 0.40,
      fontFace: 'Consolas',
      fontSize: 11,
      color: theme.secondary,
      align: 'left',
      valign: 'middle',
      margin: 0
    });
    // Port
    slide.addText(t.port, {
      x: cardX + cardW - 1.6, y: y + 0.10, w: 1.5, h: 0.40,
      fontFace: 'Consolas',
      fontSize: 11,
      bold: true,
      color: theme.accent,
      align: 'right',
      valign: 'middle',
      margin: 0
    });
    // Desc
    slide.addText(t.desc, {
      x: cardX + 0.25, y: y + 0.55, w: cardW - 0.45, h: 0.55,
      fontFace: 'Microsoft YaHei',
      fontSize: 11,
      color: theme.primary,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  });

  // Right panel: tools list
  const rx = 5.85, ry = 1.45, rw = 3.75, rh = 3.55;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: rx, y: ry, w: rw, h: rh,
    fill: { color: theme.primary },
    line: { type: 'none' },
    rectRadius: 0.08
  });

  slide.addText('MCP TOOLS', {
    x: rx + 0.25, y: ry + 0.20, w: rw - 0.50, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 10,
    bold: true,
    color: theme.light,
    charSpacing: 4,
    margin: 0
  });
  slide.addText('Action surface', {
    x: rx + 0.25, y: ry + 0.50, w: rw - 0.50, h: 0.35,
    fontFace: 'Georgia',
    fontSize: 18,
    bold: true,
    color: 'FFFFFF',
    margin: 0
  });

  const toolGroups = [
    { g: 'Protein ops',  tools: ['load_pdb', 'set_representation', 'set_color_scheme', 'apply_alphafold_plddt'] },
    { g: 'Compute ops',  tools: ['run_fep_sweep', 'run_docking', 'prep_for_fep', 'with_checkpoint'] },
    { g: 'Snapshot ops', tools: ['capture_lambda_snapshots', 'get_disappearing_atoms'] }
  ];

  toolGroups.forEach((grp, gi) => {
    const gy = ry + 1.05 + gi * 0.78;
    slide.addText(grp.g, {
      x: rx + 0.25, y: gy, w: rw - 0.50, h: 0.28,
      fontFace: 'Consolas',
      fontSize: 10,
      bold: true,
      color: theme.accent,
      charSpacing: 2,
      margin: 0
    });
    const toolText = grp.tools.map((t, i) => (
      { text: t + (i < grp.tools.length - 1 ? '  ' : ''), options: { color: 'FFFFFF', fontFace: 'Consolas' } }
    ));
    slide.addText(toolText, {
      x: rx + 0.25, y: gy + 0.28, w: rw - 0.50, h: 0.45,
      fontSize: 10,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 7, 10);
}

module.exports = { createSlide };
