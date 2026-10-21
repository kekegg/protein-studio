// Section 07 - 测试 & 现状
// Content - data viz: large pass-rate stat callout + bullet list of known issues + diff stat

const { addPageNumber, addFooter, addSlideTitle, addSectionMarker } = require('./_common.js');

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  addSlideTitle(slide, theme, '测试现状 & 已知的工程债',
    '覆盖率 161/162 通过, 1 个 flaky 视觉项;本轮 +887/-120 LOC,已知限制清单 (透明披露)');
  addSectionMarker(slide, theme, '07', 'STATUS');

  // Large stat callout on left
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.40, y: 1.45, w: 4.20, h: 3.55,
    fill: { color: theme.accent },
    line: { type: 'none' },
    rectRadius: 0.10
  });

  slide.addText('TEST PASS RATE', {
    x: 0.55, y: 1.65, w: 3.90, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 10,
    bold: true,
    color: 'C5D2E8',
    charSpacing: 4,
    margin: 0
  });

  // Big number
  slide.addText('161 / 162', {
    x: 0.55, y: 2.05, w: 3.90, h: 1.20,
    fontFace: 'Georgia',
    fontSize: 72,
    bold: true,
    color: 'FFFFFF',
    align: 'left',
    valign: 'middle',
    margin: 0
  });

  // Progress bar visual (using two rectangles)
  const pbX = 0.55, pbY = 3.30, pbW = 3.90, pbH = 0.18;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: pbX, y: pbY, w: pbW, h: pbH,
    fill: { color: 'BFD0E8' },
    line: { type: 'none' },
    rectRadius: 0.09
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: pbX, y: pbY, w: pbW * (161 / 162), h: pbH,
    fill: { color: 'FFFFFF' },
    line: { type: 'none' },
    rectRadius: 0.09
  });

  slide.addText('99.4 % pass rate  |  1 flaky visual test', {
    x: 0.55, y: 3.55, w: 3.90, h: 0.30,
    fontFace: 'Calibri',
    fontSize: 11,
    color: 'C5D2E8',
    margin: 0
  });

  slide.addText('test_ribbon_helix_is_cylindrical_not_flat', {
    x: 0.55, y: 4.10, w: 3.90, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 9,
    color: 'A8BAD3',
    margin: 0
  });

  // Side stats grid
  const sgs = [
    { l: 'src Python LOC', v: '15,868' },
    { l: 'test files', v: '14  /  4,469 L' },
    { l: 'tracked files', v: '119' }
  ];
  sgs.forEach((s, i) => {
    const sy = 3.80 + i * 0.36;
    slide.addText(
      [
        { text: s.l + '   ', options: { color: 'C5D2E8', fontSize: 10, fontFace: 'Calibri' } },
        { text: s.v, options: { color: 'FFFFFF', fontSize: 12, bold: true, fontFace: 'Georgia' } }
      ],
      {
        x: 0.55, y: sy, w: 3.90, h: 0.30,
        align: 'left',
        valign: 'middle',
        margin: 0
      }
    );
  });

  // Right side: known issues / 已知限制
  const rx = 4.90, ry = 1.45, rw = 4.70, rh = 3.55;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: rx, y: ry, w: rw, h: rh,
    fill: { color: theme.light },
    line: { type: 'none' },
    rectRadius: 0.10
  });

  slide.addText('KNOWN LIMITATIONS', {
    x: rx + 0.25, y: ry + 0.20, w: rw - 0.50, h: 0.30,
    fontFace: 'Consolas',
    fontSize: 10,
    bold: true,
    color: theme.accent,
    charSpacing: 4,
    margin: 0
  });
  slide.addText('本轮发现的工程债与已知 trade-off', {
    x: rx + 0.25, y: ry + 0.50, w: rw - 0.50, h: 0.30,
    fontFace: 'Microsoft YaHei',
    fontSize: 13,
    bold: true,
    color: theme.primary,
    margin: 0
  });

  const issues = [
    {
      tag: 'FEP-NaN',
      title: 'OpenMM addSolvent close-pair 偶发 NaN',
      detail: '三层防护已加,但根因需外部 prep (AmberTools tleap / GROMACS grompp);已在 prep_pdb_for_fep.py 提供 fix path'
    },
    {
      tag: 'VIEWPORT',
      title: 'pyvistaqt 启动路径敏感',
      detail: '用 /usr/bin/python3 启动会触发 fallback;现已改善诊断提示. 下一步在 launcher 强制 conda env python'
    },
    {
      tag: 'TEST-SLOW',
      title: 'Slow tests 在 5min timeout 上仍卡',
      detail: 'CI 需跑满 FEP sweep;应切换为分批/分线程,或预 cooked input'
    },
    {
      tag: 'PACKAGE',
      title: '跨平台打包未到位',
      detail: 'PyInstaller 试装成功但体积 ~600MB;后续考虑 conda-pack 或 dylib 裁剪'
    }
  ];

  issues.forEach((iss, i) => {
    const iy = ry + 0.95 + i * 0.62;
    slide.addText(
      [
        { text: iss.tag + '   ', options: { color: theme.accent, fontFace: 'Consolas', bold: true, fontSize: 9, charSpacing: 2 } },
        { text: iss.title, options: { color: theme.primary, fontFace: 'Microsoft YaHei', fontSize: 11, bold: true } }
      ],
      {
        x: rx + 0.25, y: iy, w: rw - 0.50, h: 0.28,
        align: 'left',
        valign: 'middle',
        margin: 0
      }
    );
    slide.addText(iss.detail, {
      x: rx + 0.25, y: iy + 0.28, w: rw - 0.50, h: 0.35,
      fontFace: 'Calibri',
      fontSize: 10,
      color: theme.secondary,
      align: 'left',
      valign: 'top',
      margin: 0
    });
  });

  addFooter(slide, theme);
  addPageNumber(slide, theme, 9, 10);
}

module.exports = { createSlide };
