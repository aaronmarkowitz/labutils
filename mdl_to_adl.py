#!/usr/bin/env python3
"""
mdl_to_adl.py - Generate MEDM .adl overview screens from Simulink .mdl models.

Parses CDS rtcds Simulink .mdl files and generates MEDM .adl screens that
mirror the block diagram layout, with clickable links to existing auto-generated
filter screens and generated popup screens for blocks without auto-screens.
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MdlBlock:
    block_type: str  # e.g. "Reference", "SubSystem", "BusSelector"
    name: str
    sid: str = ""
    tag: str = ""  # e.g. "cdsFilt", "cdsOsc", "cdsEzCaRead"
    description: str = ""
    ports_in: int = 0
    ports_out: int = 0
    position: tuple = (0, 0, 0, 0)  # x1, y1, x2, y2
    source_block: str = ""
    subsystem: Optional['MdlSystem'] = None
    background_color: str = ""
    foreground_color: str = ""
    show_name: bool = True
    drop_shadow: bool = False
    icon_shape: str = ""
    inputs_str: str = ""
    port_num: int = 1
    output_signals: str = ""


@dataclass
class MdlLine:
    src_block: str = ""
    src_port: int = 1
    dst_block: str = ""
    dst_port: int = 1
    points: list = field(default_factory=list)  # [(dx,dy), ...]
    branches: list = field(default_factory=list)  # list of MdlLine
    name: str = ""


@dataclass
class MdlAnnotation:
    text: str = ""
    position: tuple = (0, 0)  # x, y
    font_size: int = 10
    alignment: str = "left"
    drop_shadow: bool = False


@dataclass
class MdlSystem:
    name: str
    blocks: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    annotations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# ADL color map indices (standard 65-color CDS map)
# ---------------------------------------------------------------------------
# Index: hex color
# 0=ffffff, 14=000000, 15=00d800, 19=216c00, 20=fd0000, 25=5893ff,
# 33=e19015, 35=ffb0ff, 38=8b1a96, 40=a4aaff, 50=99ffff

CLR_BLACK = 14
CLR_WHITE = 0
CLR_GREEN = 15
CLR_DKGREEN = 19     # 216c00 - Simulink "darkGreen", for cdsFilt
CLR_RED = 20
CLR_BLUE = 25
CLR_CYAN = 50        # 99ffff - DAC blocks
CLR_PINK = 35        # ffb0ff - cdsOsc, cdsFiltMuxMatrix
CLR_ORANGE = 33      # e19015 - cdsParameters
CLR_LTPURPLE = 40    # a4aaff - cdsEzCaRead
CLR_DKPURPLE = 38    # 8b1a96 - cdsEzCaWrite
CLR_GREY = 5
CLR_LTGREY = 3

COLORMAP_65 = """ffffff,
ececec,
dadada,
c8c8c8,
bbbbbb,
aeaeae,
9e9e9e,
919191,
858585,
787878,
696969,
5a5a5a,
464646,
2d2d2d,
000000,
00d800,
1ebb00,
339900,
2d7f00,
216c00,
fd0000,
de1309,
be190b,
a01207,
820400,
5893ff,
597ee1,
4b6ec7,
3a5eab,
27548d,
fbf34a,
f9da3c,
eeb62b,
e19015,
cd6100,
ffb0ff,
d67fe2,
ae4ebc,
8b1a96,
610a75,
a4aaff,
8793e2,
6a73c1,
4d52a4,
343386,
c7bb6d,
b79d5c,
a47e3c,
7d5627,
58340f,
99ffff,
73dfff,
4ea5f9,
2a63e4,
0a00b8,
ebf1b5,
d4db9d,
bbc187,
a6a462,
8b8239,
73ff6b,
52da3b,
3cb420,
289315,
1a7309,"""

# Parse the palette into RGB tuples for color matching
_PALETTE_RGB = []
for _hex in COLORMAP_65.strip().replace('\n', '').split(','):
    _hex = _hex.strip()
    if _hex:
        _PALETTE_RGB.append((int(_hex[0:2], 16), int(_hex[2:4], 16), int(_hex[4:6], 16)))


def simulink_color_to_medm(color_str):
    """Map a Simulink color string to nearest MEDM palette index.

    Handles named colors (e.g. "darkGreen") and [R, G, B] float arrays
    (e.g. "[1.0, 0.49, 0.50]") where components are 0.0-1.0.
    Returns a palette index (0-64).
    """
    if not color_str:
        return CLR_WHITE

    color_str = color_str.strip()

    # Named color mapping
    named = {
        'white': CLR_WHITE, 'black': CLR_BLACK, 'red': CLR_RED,
        'green': CLR_GREEN, 'blue': CLR_BLUE, 'cyan': CLR_CYAN,
        'magenta': CLR_PINK, 'yellow': 30,
        'darkGreen': CLR_DKGREEN, 'darkgreen': CLR_DKGREEN,
        'orange': CLR_ORANGE, 'gray': CLR_GREY, 'grey': CLR_GREY,
        'lightGray': CLR_LTGREY, 'lightgray': CLR_LTGREY,
    }
    if color_str in named:
        return named[color_str]

    # Try parsing [R, G, B] float array
    m = re.match(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)', color_str)
    if m:
        r = int(float(m.group(1)) * 255)
        g = int(float(m.group(2)) * 255)
        b = int(float(m.group(3)) * 255)
        # Find nearest palette color by Euclidean distance
        best_idx = 0
        best_dist = float('inf')
        for i, (pr, pg, pb) in enumerate(_PALETTE_RGB):
            d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    return CLR_WHITE


# ---------------------------------------------------------------------------
# MdlParser
# ---------------------------------------------------------------------------

class MdlParser:
    """Parse Simulink .mdl text format into MdlSystem trees."""

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.model_name = ""
        self.ifo = ""
        self.rate = ""
        self.dcuid = ""
        self.host = ""

    def parse(self) -> MdlSystem:
        """Parse the full .mdl file, return the top-level MdlSystem."""
        root_system = None
        self._skip_whitespace()

        if not self.text.startswith("Model"):
            raise ValueError("Not a valid .mdl file: missing Model block")

        m = re.search(r'Name\s+"([^"]+)"', self.text[:500])
        if m:
            self.model_name = m.group(1)

        root_system = self._find_and_parse_top_system()
        return root_system

    def _find_and_parse_top_system(self) -> MdlSystem:
        """Find and parse the top-level System block."""
        pattern = re.compile(r'^  System \{', re.MULTILINE)
        m = pattern.search(self.text)
        if not m:
            raise ValueError("No top-level System block found")

        self.pos = m.start()
        return self._parse_system_block()

    def _parse_system_block(self) -> MdlSystem:
        """Parse a System { ... } block starting at self.pos."""
        self.pos = self.text.index('{', self.pos) + 1
        system = MdlSystem(name="")

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break

            if self.text[self.pos] == '}':
                self.pos += 1
                break

            key, value = self._read_key_value()

            if key == 'Name':
                system.name = value
            elif key == 'Block':
                block = self._parse_block()
                if block:
                    system.blocks.append(block)
            elif key == 'Line':
                line = self._parse_line()
                if line:
                    system.lines.append(line)
            elif key == 'Annotation':
                ann = self._parse_annotation()
                if ann:
                    system.annotations.append(ann)
            # Ignore other keys

        # Extract cdsParameters info
        for block in system.blocks:
            if block.tag == 'cdsParameters':
                self._extract_params(block.name)

        return system

    def _extract_params(self, name):
        """Extract IFO, rate, etc from cdsParameters block name."""
        for part in name.replace('\\n', '\n').split('\n'):
            part = part.strip()
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k == 'ifo':
                    self.ifo = v
                elif k == 'rate':
                    self.rate = v
                elif k == 'dcuid':
                    self.dcuid = v
                elif k == 'host':
                    self.host = v

    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def _read_key_value(self):
        """Read a key and optional value. Handles braces for block starts."""
        self._skip_whitespace()

        if self.text[self.pos] == '"':
            end = self.text.index('"', self.pos + 1)
            key = self.text[self.pos + 1:end]
            self.pos = end + 1
        else:
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] not in ' \t\n\r{}':
                self.pos += 1
            key = self.text[start:self.pos]

        self._skip_whitespace()

        if self.pos < len(self.text) and self.text[self.pos] == '{':
            return key, None

        value = self._read_value()
        return key, value

    def _read_value(self):
        """Read a value (string, number, array, or multi-line string)."""
        self._skip_whitespace()
        if self.pos >= len(self.text):
            return ""

        c = self.text[self.pos]

        if c == '"':
            return self._read_string()
        elif c == '[':
            return self._read_array()
        elif c == '{':
            return None
        else:
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] != '\n':
                self.pos += 1
            return self.text[start:self.pos].strip()

    def _read_string(self):
        """Read a quoted string, handling multi-line concatenation."""
        result = []
        while self.pos < len(self.text) and self.text[self.pos] == '"':
            self.pos += 1
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] != '"':
                if self.text[self.pos] == '\\' and self.pos + 1 < len(self.text):
                    self.pos += 2
                else:
                    self.pos += 1
            segment = self.text[start:self.pos]
            self.pos += 1
            result.append(segment)
            self._skip_whitespace()

        return ''.join(result)

    def _read_array(self):
        """Read a bracketed array like [1, 2] or [0.5, 0.5, 0.5, 0.5]."""
        start = self.pos
        depth = 0
        while self.pos < len(self.text):
            if self.text[self.pos] == '[':
                depth += 1
            elif self.text[self.pos] == ']':
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    return self.text[start:self.pos]
            self.pos += 1
        return self.text[start:self.pos]

    def _skip_brace_block(self):
        """Skip a { ... } block, handling nesting."""
        self._skip_whitespace()
        if self.pos < len(self.text) and self.text[self.pos] == '{':
            depth = 1
            self.pos += 1
            in_string = False
            while self.pos < len(self.text) and depth > 0:
                c = self.text[self.pos]
                if in_string:
                    if c == '\\':
                        self.pos += 1
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                self.pos += 1

    def _parse_block(self) -> Optional[MdlBlock]:
        """Parse a Block { ... } element."""
        self._skip_whitespace()
        if self.text[self.pos] != '{':
            return None
        self.pos += 1

        block = MdlBlock(block_type="", name="")

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            if self.text[self.pos] == '}':
                self.pos += 1
                break

            key, value = self._read_key_value()

            if key == 'BlockType':
                block.block_type = value
            elif key == 'Name':
                block.name = value if value else ""
            elif key == 'SID':
                block.sid = value if value else ""
            elif key == 'Tag':
                block.tag = value if value else ""
            elif key == 'Description':
                block.description = value if value else ""
            elif key == 'Ports':
                block.ports_in, block.ports_out = self._parse_ports(value)
            elif key == 'Position':
                block.position = self._parse_position(value)
            elif key == 'SourceBlock':
                block.source_block = value if value else ""
            elif key == 'BackgroundColor':
                block.background_color = value if value else ""
            elif key == 'ForegroundColor':
                block.foreground_color = value if value else ""
            elif key == 'ShowName':
                block.show_name = (value.strip().lower() != 'off') if value else True
            elif key == 'DropShadow':
                block.drop_shadow = (value.strip().lower() == 'on') if value else False
            elif key == 'IconShape':
                block.icon_shape = value if value else ""
            elif key == 'Inputs':
                block.inputs_str = value if value else ""
            elif key == 'Port':
                if value is not None and isinstance(value, str) and value.strip().isdigit():
                    block.port_num = int(value.strip())
                elif value is None:
                    self._skip_brace_block()
            elif key == 'OutputSignals':
                block.output_signals = value if value else ""
            elif key == 'System':
                block.subsystem = self._parse_system_block()
            elif value is None:
                self._skip_brace_block()

        return block

    def _parse_line(self) -> Optional[MdlLine]:
        """Parse a Line { ... } element."""
        self._skip_whitespace()
        if self.text[self.pos] != '{':
            return None
        self.pos += 1

        line = MdlLine()

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            if self.text[self.pos] == '}':
                self.pos += 1
                break

            key, value = self._read_key_value()

            if key == 'Name':
                line.name = value if value else ""
            elif key == 'SrcBlock':
                line.src_block = value if value else ""
            elif key == 'SrcPort':
                line.src_port = int(value) if value else 1
            elif key == 'DstBlock':
                line.dst_block = value if value else ""
            elif key == 'DstPort':
                line.dst_port = int(value) if value else 1
            elif key == 'Points':
                line.points = self._parse_points(value)
            elif key == 'Branch':
                branch = self._parse_branch()
                if branch:
                    line.branches.append(branch)
            elif value is None:
                self._skip_brace_block()

        return line

    def _parse_branch(self) -> Optional[MdlLine]:
        """Parse a Branch { ... } inside a Line."""
        self._skip_whitespace()
        if self.text[self.pos] != '{':
            return None
        self.pos += 1

        branch = MdlLine()

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            if self.text[self.pos] == '}':
                self.pos += 1
                break

            key, value = self._read_key_value()

            if key == 'Name':
                branch.name = value if value else ""
            elif key == 'DstBlock':
                branch.dst_block = value if value else ""
            elif key == 'DstPort':
                branch.dst_port = int(value) if value else 1
            elif key == 'Points':
                branch.points = self._parse_points(value)
            elif key == 'Branch':
                sub = self._parse_branch()
                if sub:
                    branch.branches.append(sub)
            elif value is None:
                self._skip_brace_block()

        return branch

    def _parse_annotation(self) -> Optional[MdlAnnotation]:
        """Parse an Annotation { ... } element."""
        self._skip_whitespace()
        if self.pos >= len(self.text) or self.text[self.pos] != '{':
            return None
        self.pos += 1

        ann = MdlAnnotation()

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            if self.text[self.pos] == '}':
                self.pos += 1
                break

            key, value = self._read_key_value()

            if key == 'Name':
                ann.text = value if value else ""
            elif key == 'Position':
                pos = self._parse_position_2d(value)
                ann.position = pos
            elif key == 'FontSize':
                try:
                    ann.font_size = int(value)
                except (ValueError, TypeError):
                    pass
            elif key == 'HorizontalAlignment':
                ann.alignment = value if value else "left"
            elif key == 'DropShadow':
                ann.drop_shadow = (value.strip().lower() == 'on') if value else False
            elif value is None:
                self._skip_brace_block()

        if ann.text:
            return ann
        return None

    def _parse_position_2d(self, value):
        """Parse Position: [x, y] or [x1, y1, x2, y2] (use top-left)."""
        if not value:
            return (0, 0)
        inner = value.strip('[]').strip()
        parts = [x.strip() for x in inner.split(',')]
        if len(parts) >= 2:
            return (int(float(parts[0])), int(float(parts[1])))
        return (0, 0)

    def _parse_ports(self, value):
        """Parse Ports field: [in, out] or [n] (n inputs only)."""
        if not value or value == '[]':
            return 0, 0
        inner = value.strip('[]').strip()
        if not inner:
            return 0, 0
        parts = [x.strip() for x in inner.split(',')]
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            return int(parts[0]), 0
        return 0, 0

    def _parse_position(self, value):
        """Parse Position: [x1, y1, x2, y2]."""
        if not value:
            return (0, 0, 0, 0)
        inner = value.strip('[]').strip()
        parts = [x.strip() for x in inner.split(',')]
        if len(parts) == 4:
            return tuple(int(float(x)) for x in parts)
        return (0, 0, 0, 0)

    def _parse_points(self, value):
        """Parse Points: [dx, dy; dx, dy] relative waypoints."""
        if not value:
            return []
        inner = value.strip('[]').strip()
        if not inner:
            return []
        points = []
        for segment in inner.split(';'):
            parts = [x.strip() for x in segment.strip().split(',')]
            if len(parts) == 2:
                try:
                    points.append((int(float(parts[0])), int(float(parts[1]))))
                except ValueError:
                    pass
        return points


# ---------------------------------------------------------------------------
# AdlWriter
# ---------------------------------------------------------------------------

class AdlWriter:
    """Low-level ADL format writer."""

    def __init__(self):
        self.lines = []

    def write_header(self, name, width, height):
        self.lines.append('')
        self.lines.append('file {')
        self.lines.append(f'\tname="{name}"')
        self.lines.append('\tversion=030117')
        self.lines.append('}')
        self.lines.append('display {')
        self.lines.append('\tobject {')
        self.lines.append('\t\tx=100')
        self.lines.append('\t\ty=100')
        self.lines.append(f'\t\twidth={width}')
        self.lines.append(f'\t\theight={height}')
        self.lines.append('\t}')
        self.lines.append(f'\tclr={CLR_BLACK}')
        self.lines.append(f'\tbclr={CLR_LTGREY}')
        self.lines.append('\tcmap=""')
        self.lines.append('\tgridSpacing=5')
        self.lines.append('\tgridOn=0')
        self.lines.append('\tsnapToGrid=0')
        self.lines.append('}')
        self._write_colormap()

    def _write_colormap(self):
        self.lines.append('"color map" {')
        self.lines.append('\tncolors=65')
        self.lines.append('\tcolors {')
        for color_line in COLORMAP_65.strip().split('\n'):
            self.lines.append(f'\t\t{color_line.strip()}')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_related_display(self, x, y, w, h, label, target_adl, clr=CLR_BLACK, bclr=CLR_LTGREY, button_label=None):
        face_label = button_label if button_label is not None else label
        self.lines.append('"related display" {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append('\tdisplay[0] {')
        self.lines.append(f'\t\tlabel="{label}"')
        self.lines.append(f'\t\tname="{target_adl}"')
        self.lines.append('\t}')
        self.lines.append(f'\tclr={clr}')
        self.lines.append(f'\tbclr={bclr}')
        self.lines.append(f'\tlabel="{face_label}"')
        self.lines.append('}')

    def write_text(self, x, y, w, h, text, clr=CLR_BLACK, align="horiz. centered"):
        self.lines.append('text {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={clr}')
        self.lines.append('\t}')
        self.lines.append(f'\ttextix="{text}"')
        self.lines.append(f'\talign="{align}"')
        self.lines.append('}')

    def write_rectangle(self, x, y, w, h, fill_clr=CLR_LTGREY, line_clr=CLR_BLACK):
        self.lines.append('rectangle {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={line_clr}')
        self.lines.append(f'\t\tfill="outline"')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_filled_rectangle(self, x, y, w, h, fill_clr, border_clr=CLR_BLACK):
        """Emit a solid fill rectangle followed by an outline border rectangle."""
        # Solid fill
        self.lines.append('rectangle {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={fill_clr}')
        self.lines.append('\t}')
        self.lines.append('}')
        # Outline border
        self.lines.append('rectangle {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={border_clr}')
        self.lines.append(f'\t\tfill="outline"')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_oval(self, x, y, w, h, fill_clr, border_clr=CLR_BLACK):
        """Emit a filled oval followed by an outline oval for the border."""
        # Solid fill
        self.lines.append('oval {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={fill_clr}')
        self.lines.append('\t}')
        self.lines.append('}')
        # Outline border
        self.lines.append('oval {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={border_clr}')
        self.lines.append(f'\t\tfill="outline"')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_polyline(self, points, clr=CLR_BLACK, width=1):
        if len(points) < 2:
            return
        self.lines.append('polyline {')
        self.lines.append('\tobject {')
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        self.lines.append(f'\t\tx={x0}')
        self.lines.append(f'\t\ty={y0}')
        self.lines.append(f'\t\twidth={max(x1 - x0, 1)}')
        self.lines.append(f'\t\theight={max(y1 - y0, 1)}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={clr}')
        if width > 1:
            self.lines.append(f'\t\twidth={width}')
        self.lines.append('\t}')
        self.lines.append('\tpoints {')
        for px, py in points:
            self.lines.append(f'\t\t({px},{py})')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_polygon(self, points, clr=CLR_BLACK, fill_clr=None):
        """Emit a filled polygon (like polyline but closed and filled)."""
        if len(points) < 3:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        if fill_clr is not None:
            self.lines.append('polygon {')
            self.lines.append('\tobject {')
            self.lines.append(f'\t\tx={x0}')
            self.lines.append(f'\t\ty={y0}')
            self.lines.append(f'\t\twidth={max(x1 - x0, 1)}')
            self.lines.append(f'\t\theight={max(y1 - y0, 1)}')
            self.lines.append('\t}')
            self.lines.append(f'\t"basic attribute" {{')
            self.lines.append(f'\t\tclr={fill_clr}')
            self.lines.append('\t}')
            self.lines.append('\tpoints {')
            for px, py in points:
                self.lines.append(f'\t\t({px},{py})')
            self.lines.append('\t}')
            self.lines.append('}')
        # Outline
        self.lines.append('polygon {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x0}')
        self.lines.append(f'\t\ty={y0}')
        self.lines.append(f'\t\twidth={max(x1 - x0, 1)}')
        self.lines.append(f'\t\theight={max(y1 - y0, 1)}')
        self.lines.append('\t}')
        self.lines.append(f'\t"basic attribute" {{')
        self.lines.append(f'\t\tclr={clr}')
        self.lines.append(f'\t\tfill="outline"')
        self.lines.append('\t}')
        self.lines.append('\tpoints {')
        for px, py in points:
            self.lines.append(f'\t\t({px},{py})')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_arrow(self, x, y, direction, size=5, clr=CLR_BLACK):
        """Draw a small filled triangle arrow head at (x,y)."""
        if direction == 'right':
            points = [(x - size, y - size // 2), (x, y), (x - size, y + size // 2)]
        elif direction == 'left':
            points = [(x + size, y - size // 2), (x, y), (x + size, y + size // 2)]
        elif direction == 'down':
            points = [(x - size // 2, y - size), (x, y), (x + size // 2, y - size)]
        elif direction == 'up':
            points = [(x - size // 2, y + size), (x, y), (x + size // 2, y + size)]
        else:
            return
        self.write_polygon(points, clr=clr, fill_clr=clr)

    def write_text_update(self, x, y, w, h, channel, clr=CLR_BLACK, bclr=CLR_WHITE):
        self.lines.append('"text update" {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\tmonitor {{')
        self.lines.append(f'\t\tchan="{channel}"')
        self.lines.append(f'\t\tclr={clr}')
        self.lines.append(f'\t\tbclr={bclr}')
        self.lines.append('\t}')
        self.lines.append('}')

    def write_text_entry(self, x, y, w, h, channel, clr=CLR_BLACK, bclr=CLR_WHITE):
        self.lines.append('"text entry" {')
        self.lines.append('\tobject {')
        self.lines.append(f'\t\tx={x}')
        self.lines.append(f'\t\ty={y}')
        self.lines.append(f'\t\twidth={w}')
        self.lines.append(f'\t\theight={h}')
        self.lines.append('\t}')
        self.lines.append(f'\tcontrol {{')
        self.lines.append(f'\t\tchan="{channel}"')
        self.lines.append(f'\t\tclr={clr}')
        self.lines.append(f'\t\tbclr={bclr}')
        self.lines.append('\t}')
        self.lines.append('}')

    def get_text(self):
        return '\n'.join(self.lines) + '\n'


# ---------------------------------------------------------------------------
# AdlGenerator
# ---------------------------------------------------------------------------

class AdlGenerator:
    """Orchestrates generation of overview .adl screens from parsed MdlSystems."""

    def __init__(self, model_name, ifo, medm_dir, output_dir, scale=1.0,
                 host="", rate="", dcuid=""):
        self.model_name = model_name
        self.ifo = ifo
        self.prefix = model_name.upper() if model_name else ""
        self.medm_dir = medm_dir
        self.output_dir = output_dir
        self.scale = scale
        self.host = host
        self.rate = rate
        self.dcuid = dcuid
        self.generated_files = []

    def generate_all(self, system: MdlSystem):
        """Generate overview screens for the top-level system and all subsystems."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._generate_system_screen(system, self.prefix, [])

    def _generate_system_screen(self, system: MdlSystem, prefix: str, path: list):
        """Generate an overview .adl for one system level."""
        filename = f"{prefix}_OVERVIEW.adl"
        filepath = os.path.join(self.output_dir, filename)

        block_map = {b.name: b for b in system.blocks}

        # Compute coordinate transform - include blocks and annotations
        padding = 50
        positions = [b.position for b in system.blocks if b.position != (0, 0, 0, 0)]
        if not positions:
            return

        min_x = min(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_x = max(p[2] for p in positions)
        max_y = max(p[3] for p in positions)

        # Include annotation positions in bounds (both min and max)
        for ann in system.annotations:
            ax, ay = ann.position
            if ax != 0 or ay != 0:
                text_w = len(ann.text.split('\\n')[0]) * 7 if ann.text else 0
                text_lines = ann.text.count('\\n') + 1
                text_h = text_lines * (ann.font_size + 2) if ann.font_size else 14
                min_x = min(min_x, ax)
                min_y = min(min_y, ay)
                max_x = max(max_x, ax + text_w)
                max_y = max(max_y, ay + text_h)

        # Add extra space at bottom for labels below blocks
        max_y += 18

        def tx(x):
            return int((x - min_x + padding) * self.scale)

        def ty(y):
            return int((y - min_y + padding) * self.scale)

        display_w = int((max_x - min_x + 2 * padding) * self.scale)
        display_h = int((max_y - min_y + 2 * padding) * self.scale)

        title_h = 30
        display_h += title_h

        def ty_off(y):
            return ty(y) + title_h

        writer = AdlWriter()
        writer.write_header(filepath, display_w, display_h)

        # Title
        title_text = prefix.replace('_', ' ') + " Overview"
        writer.write_text(0, 2, display_w, 25, title_text, clr=CLR_BLACK)

        # Render blocks
        for block in system.blocks:
            x1, y1, x2, y2 = block.position
            if (x1, y1, x2, y2) == (0, 0, 0, 0):
                continue

            bx = tx(x1)
            by = ty_off(y1)
            bw = int((x2 - x1) * self.scale)
            bh = int((y2 - y1) * self.scale)
            bw = max(bw, 20)
            bh = max(bh, 15)

            display_name = block.name.replace('\\n', ' ').replace('\n', ' ')

            tag = block.tag
            bt = block.block_type

            # Determine short label for button and whether to show name below
            show_label = block.show_name

            if tag == 'cdsFilt':
                block_path = '_'.join(path + [block.name])
                target = os.path.join(self.medm_dir,
                                      f"{self.prefix}_{block_path}.adl")
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             target, CLR_BLACK, CLR_DKGREEN,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsFiltMuxMatrix':
                target = os.path.join(self.medm_dir,
                                      f"{self.prefix}_{block.name}.adl")
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             target, CLR_BLACK, CLR_PINK,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsRampMuxMatrix':
                block_path = '_'.join(path + [block.name])
                target = os.path.join(self.medm_dir,
                                      f"{self.prefix}_{block_path}.adl")
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             target, CLR_BLACK, CLR_GREEN,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsOsc':
                popup_gen = PopupGenerator(self.prefix, path, self.output_dir)
                popup_file = popup_gen.generate_osc_popup(block)
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             popup_file, CLR_BLACK, CLR_PINK,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsEzCaRead':
                popup_gen = PopupGenerator(self.prefix, path, self.output_dir)
                popup_file = popup_gen.generate_ezca_read_popup(block)
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             popup_file, CLR_BLACK, CLR_LTPURPLE,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsEzCaWrite':
                popup_gen = PopupGenerator(self.prefix, path, self.output_dir)
                popup_file = popup_gen.generate_ezca_write_popup(block)
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             popup_file, CLR_BLACK, CLR_DKPURPLE,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsAtan2':
                popup_gen = PopupGenerator(self.prefix, path, self.output_dir)
                popup_file = popup_gen.generate_atan2_popup(block)
                # White box with atan2 text, clickable overlay
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             popup_file, CLR_BLACK, CLR_WHITE,
                                             button_label="")
                writer.write_text(bx, by + 2, bw, 14, "atan2(y,x)",
                                  clr=CLR_BLACK)
                if block.ports_in >= 2:
                    py1 = by + bh * 1 // 3
                    py2 = by + bh * 2 // 3
                    writer.write_text(bx + 2, py1 - 6, 12, 12, "y",
                                      clr=CLR_BLACK, align="horiz. left")
                    writer.write_text(bx + 2, py2 - 6, 12, 12, "x",
                                      clr=CLR_BLACK, align="horiz. left")
                if block.ports_out >= 1:
                    py_out = by + bh // 2
                    writer.write_text(bx + bw - 20, py_out - 6, 18, 12,
                                      "out", clr=CLR_BLACK,
                                      align="horiz. right")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsSqrt':
                popup_gen = PopupGenerator(self.prefix, path, self.output_dir)
                popup_file = popup_gen.generate_sqrt_popup(block)
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             popup_file, CLR_BLACK, CLR_GREY,
                                             button_label="")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif bt == 'SubSystem' and block.subsystem:
                sub_prefix = f"{prefix}_{block.name}"
                sub_overview = os.path.join(self.output_dir,
                                            f"{sub_prefix}_OVERVIEW.adl")
                writer.write_related_display(bx, by, bw, bh, block.name,
                                             sub_overview, CLR_WHITE, CLR_BLUE,
                                             button_label="")
                # Port name labels on subsystem block face
                inports = sorted([b for b in block.subsystem.blocks
                                   if b.block_type == 'Inport'],
                                  key=lambda b: b.port_num)
                outports = sorted([b for b in block.subsystem.blocks
                                    if b.block_type == 'Outport'],
                                   key=lambda b: b.port_num)
                # Port name labels OUTSIDE the button (Motif buttons cover overlaid text)
                if inports:
                    n_in = len(inports)
                    for i, inp in enumerate(inports):
                        py = by + bh * (i + 1) // (n_in + 1)
                        pname = inp.name.replace('\\n', ' ')
                        label_w = max(len(pname) * 7, 20)
                        writer.write_text(bx - label_w - 2, py - 6,
                                          label_w, 12,
                                          pname, clr=CLR_BLACK,
                                          align="horiz. right")
                if outports:
                    n_out = len(outports)
                    for i, outp in enumerate(outports):
                        py = by + bh * (i + 1) // (n_out + 1)
                        pname = outp.name.replace('\\n', ' ')
                        label_w = max(len(pname) * 7, 20)
                        writer.write_text(bx + bw + 2, py - 6,
                                          label_w, 12,
                                          pname, clr=CLR_BLACK,
                                          align="horiz. left")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)
                self._generate_system_screen(block.subsystem, sub_prefix,
                                             path + [block.name])

            elif tag == 'cdsAdc':
                writer.write_filled_rectangle(bx, by, bw, bh, CLR_WHITE)
                writer.write_text(bx, by, bw, bh, display_name, clr=CLR_RED)
                if block.description:
                    desc = block.description.replace('\\n', ' ')
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      desc, clr=CLR_RED)
                elif show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_RED)

            elif bt == 'Reference' and 'dac' in block.source_block.lower():
                writer.write_filled_rectangle(bx, by, bw, bh, CLR_CYAN)
                writer.write_text(bx, by, bw, min(bh, 20), display_name,
                                  clr=CLR_BLACK)
                # Port labels inside DAC block
                if block.ports_in > 0:
                    for p in range(block.ports_in):
                        py = by + bh * (p + 1) // (block.ports_in + 1)
                        writer.write_text(bx + 2, py - 6, 30, 12,
                                          f"In{p}", clr=CLR_BLACK,
                                          align="horiz. left")
                if show_label:
                    writer.write_text(bx, by + bh + 2, bw, 14,
                                      display_name, clr=CLR_BLACK)

            elif tag == 'cdsParameters':
                writer.write_filled_rectangle(bx, by, bw, bh, CLR_ORANGE)
                # Render param text BELOW the box
                param_y = by + bh + 2
                line_h = 14
                param_text = block.name.replace('\\n', '\n')
                for pline in param_text.split('\n'):
                    pline = pline.strip()
                    if pline:
                        writer.write_text(bx + 2, param_y,
                                          max(len(pline) * 7, bw), line_h,
                                          pline, clr=CLR_BLACK,
                                          align="horiz. left")
                        param_y += line_h
                writer.write_text(bx, param_y, bw, line_h,
                                  "cdsParameters", clr=CLR_BLACK)

            elif bt in ('BusSelector', 'Demux', 'Mux'):
                writer.write_filled_rectangle(bx, by, bw, bh, CLR_BLACK)
                # BusSelector output signal labels
                if bt == 'BusSelector' and block.output_signals:
                    signals = [s.strip() for s in
                               block.output_signals.split(',')]
                    n_sig = len(signals)
                    for i, sig in enumerate(signals):
                        if sig:
                            py = by + bh * (i + 1) // (n_sig + 1)
                            tw = max(len(sig) * 7, 20)
                            writer.write_text(
                                bx + bw + 2, py - 6, tw, 12, sig,
                                clr=CLR_BLACK, align="horiz. left")

            elif bt == 'Sum':
                if block.icon_shape.lower() == 'round' if block.icon_shape else False:
                    # Circular sum block
                    size = min(bw, bh)
                    cx_off = bx + (bw - size) // 2
                    cy_off = by + (bh - size) // 2
                    writer.write_oval(cx_off, cy_off, size, size,
                                      CLR_WHITE, CLR_BLACK)
                    # Render +/- signs from inputs_str at input port positions
                    signs = block.inputs_str.strip().strip('"')
                    sign_chars = [c for c in signs if c in '+-']
                    if sign_chars:
                        n_signs = len(sign_chars)
                        for i, sc in enumerate(sign_chars):
                            # Position signs inside circle near left edge
                            sy = cy_off + size * (i + 1) // (n_signs + 1)
                            writer.write_text(cx_off + 2, sy - 6, 12, 12,
                                              sc, clr=CLR_BLACK)
                else:
                    writer.write_filled_rectangle(bx, by, bw, bh, CLR_WHITE)
                    signs = block.inputs_str.strip().strip('"') if block.inputs_str else "+"
                    sign_chars = [c for c in signs if c in '+-']
                    label = ''.join(sign_chars) if sign_chars else "+"
                    writer.write_text(bx, by, bw, bh, label, clr=CLR_BLACK)

            elif bt == 'Product':
                writer.write_filled_rectangle(bx, by, bw, bh, CLR_WHITE)
                signs = block.inputs_str.strip().strip('"') if block.inputs_str else "*"
                sign_chars = [c for c in signs if c in '*/']
                if sign_chars:
                    n_signs = len(sign_chars)
                    for i, sc in enumerate(sign_chars):
                        sy = by + bh * (i + 1) // (n_signs + 1)
                        display_sign = 'x' if sc == '*' else '/'
                        writer.write_text(bx + 2, sy - 6, 12, 12,
                                          display_sign, clr=CLR_BLACK)
                else:
                    writer.write_text(bx, by, bw, bh, "x", clr=CLR_BLACK)

            elif bt == 'Inport':
                oval_size = min(bh, 20)
                oval_x = bx + (bw - oval_size) // 2
                oval_y = by + (bh - oval_size) // 2
                writer.write_oval(oval_x, oval_y, oval_size, oval_size,
                                  CLR_WHITE, CLR_BLACK)
                writer.write_text(oval_x, oval_y, oval_size, oval_size,
                                  str(block.port_num), clr=CLR_BLACK)
                writer.write_text(bx, by + bh + 2, bw, 14,
                                  display_name, clr=CLR_BLACK)

            elif bt == 'Outport':
                oval_size = min(bh, 20)
                oval_x = bx + (bw - oval_size) // 2
                oval_y = by + (bh - oval_size) // 2
                writer.write_oval(oval_x, oval_y, oval_size, oval_size,
                                  CLR_WHITE, CLR_BLACK)
                writer.write_text(oval_x, oval_y, oval_size, oval_size,
                                  str(block.port_num), clr=CLR_BLACK)
                writer.write_text(bx, by + bh + 2, bw, 14,
                                  display_name, clr=CLR_BLACK)

            elif bt == 'Ground':
                cy = by + bh // 2
                bar_spacing = max(bw // 6, 3)
                # Three vertical crossbars on left: shortest far-left, longest near stem
                for i in range(3):
                    lh = bh * (i + 1) // 3
                    lx = bx + i * bar_spacing
                    writer.write_polyline([(lx, cy - lh // 2),
                                           (lx, cy + lh // 2)], CLR_BLACK)
                # Horizontal stem from longest bar to right edge
                writer.write_polyline([(bx + 2 * bar_spacing, cy),
                                       (bx + bw, cy)], CLR_BLACK)

            elif bt == 'Terminator':
                # Right-pointing filled triangle
                points = [(bx, by), (bx + bw, by + bh // 2), (bx, by + bh)]
                writer.write_polygon(points, clr=CLR_BLACK, fill_clr=CLR_GREY)

            else:
                writer.write_rectangle(bx, by, bw, bh, CLR_WHITE, CLR_BLACK)
                if bw > 15:
                    writer.write_text(bx, by, bw, min(bh, 15), display_name,
                                      clr=CLR_BLACK)

        # Render annotations
        for ann in system.annotations:
            ax, ay = ann.position
            if ax == 0 and ay == 0:
                continue
            sx = tx(ax)
            sy = ty_off(ay)
            # Split on literal \n for multi-line text
            text = ann.text.replace('\\n', '\n')
            text_lines = text.split('\n')
            line_h = max(ann.font_size + 2, 14) if ann.font_size else 14
            adl_align = "horiz. left"
            if ann.alignment and 'center' in ann.alignment.lower():
                adl_align = "horiz. centered"
            elif ann.alignment and 'right' in ann.alignment.lower():
                adl_align = "horiz. right"
            for i, tline in enumerate(text_lines):
                if tline.strip():
                    tw = max(len(tline) * 7, 20)
                    writer.write_text(sx, sy + i * line_h, tw, line_h,
                                      tline, clr=CLR_BLACK, align=adl_align)
            # Box around #DAQ annotations
            if ann.text.startswith('#DAQ'):
                total_h = len(text_lines) * line_h
                max_tw = max((len(tl) * 7 for tl in text_lines
                              if tl.strip()), default=20)
                writer.write_rectangle(sx - 3, sy - 3, max_tw + 6,
                                       total_h + 6, CLR_BLACK, CLR_BLACK)

        # Render lines
        for line in system.lines:
            self._render_line(writer, line, block_map, tx, ty_off)

        # Write file
        with open(filepath, 'w') as f:
            f.write(writer.get_text())
        self.generated_files.append(filepath)
        print(f"  Generated: {filepath}")

    def _get_port_position(self, block, port_num, is_output, tx, ty):
        """Calculate screen position for a port on a block."""
        x1, y1, x2, y2 = block.position
        bh = y2 - y1

        if is_output:
            total = max(block.ports_out, 1)
            px = x2
        else:
            total = max(block.ports_in, 1)
            px = x1

        spacing = bh / (total + 1)
        py = y1 + spacing * port_num

        return tx(int(px)), ty(int(py))

    def _draw_arrow_at_end(self, writer, points):
        """Draw an arrow head at the end of a polyline."""
        if len(points) < 2:
            return
        px, py = points[-2]
        ex, ey = points[-1]
        dx = ex - px
        dy = ey - py
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        elif abs(dy) > 0:
            direction = 'down' if dy > 0 else 'up'
        else:
            return
        writer.write_arrow(ex, ey, direction, 5, CLR_BLACK)

    def _render_line(self, writer, line, block_map, tx, ty):
        """Render a line (connection) between blocks."""
        src = block_map.get(line.src_block)
        if not src:
            return

        src_x, src_y = self._get_port_position(src, line.src_port, True, tx, ty)

        if line.branches:
            branch_point_x, branch_point_y = src_x, src_y

            if line.points:
                cx, cy = src_x, src_y
                for dx, dy in line.points:
                    nx, ny = cx + int(dx * self.scale), cy + int(dy * self.scale)
                    writer.write_polyline([(cx, cy), (nx, ny)], CLR_BLACK)
                    cx, cy = nx, ny
                branch_point_x, branch_point_y = cx, cy

            suppress = src.block_type in ('BusSelector', 'Demux', 'Mux')
            for branch in line.branches:
                self._render_branch(writer, branch, block_map,
                                    branch_point_x, branch_point_y, tx, ty,
                                    suppress_wire_labels=suppress)
        elif line.dst_block:
            dst = block_map.get(line.dst_block)
            if not dst:
                return
            dst_x, dst_y = self._get_port_position(dst, line.dst_port, False, tx, ty)

            if line.points:
                points = [(src_x, src_y)]
                cx, cy = src_x, src_y
                for dx, dy in line.points:
                    cx += int(dx * self.scale)
                    cy += int(dy * self.scale)
                    points.append((cx, cy))
                # Orthogonal final connection — snap small gaps to avoid jogs
                dy_gap = abs(cy - dst_y)
                dx_gap = abs(cx - dst_x)
                if cy != dst_y and dy_gap <= 20 and len(points) >= 2:
                    # Snap y of last waypoint to destination y
                    points[-1] = (points[-1][0], dst_y)
                    points.append((dst_x, dst_y))
                elif cx != dst_x and cy != dst_y:
                    # Large gap: vertical first, horizontal last
                    points.append((cx, dst_y))
                    points.append((dst_x, dst_y))
                else:
                    points.append((dst_x, dst_y))
                writer.write_polyline(points, CLR_BLACK)
                self._draw_arrow_at_end(writer, points)
            else:
                # L-shaped route
                if src_y == dst_y:
                    points = [(src_x, src_y), (dst_x, dst_y)]
                elif src_x == dst_x:
                    points = [(src_x, src_y), (dst_x, dst_y)]
                else:
                    mid_x = (src_x + dst_x) // 2
                    points = [(src_x, src_y), (mid_x, src_y),
                              (mid_x, dst_y), (dst_x, dst_y)]
                writer.write_polyline(points, CLR_BLACK)
                self._draw_arrow_at_end(writer, points)

        # Signal label on wire (suppress if source is BusSelector/Demux/Mux)
        suppress_wire_labels = src.block_type in ('BusSelector', 'Demux', 'Mux')
        if line.name and not suppress_wire_labels:
            label_text = line.name.strip('<>').strip('"')
            if label_text:
                lx = src_x + 5
                ly = src_y - 14
                lw = max(len(label_text) * 7, 20)
                writer.write_text(lx, ly, lw, 12, label_text,
                                  clr=CLR_BLACK, align="horiz. left")

    def _render_branch(self, writer, branch, block_map, start_x, start_y, tx, ty,
                       suppress_wire_labels=False):
        """Render a branch from a branch point to its destination."""
        cx, cy = start_x, start_y

        # Follow branch waypoints to compute final position
        if branch.points:
            for dx, dy in branch.points:
                nx, ny = cx + int(dx * self.scale), cy + int(dy * self.scale)
                cx, cy = nx, ny

        if branch.branches:
            # Nested branches: draw to branch point via waypoints, then recurse
            if branch.points:
                # Build proper orthogonal segments through waypoints
                points = [(start_x, start_y)]
                px, py = start_x, start_y
                for dx, dy in branch.points:
                    px += int(dx * self.scale)
                    py += int(dy * self.scale)
                    points.append((px, py))
                writer.write_polyline(points, CLR_BLACK)
            for sub in branch.branches:
                self._render_branch(writer, sub, block_map, cx, cy, tx, ty,
                                    suppress_wire_labels=suppress_wire_labels)
        elif branch.dst_block:
            dst = block_map.get(branch.dst_block)
            if not dst:
                return
            dst_x, dst_y = self._get_port_position(dst, branch.dst_port,
                                                    False, tx, ty)
            if branch.points:
                # Multi-segment via waypoints
                points = [(start_x, start_y)]
                px, py = start_x, start_y
                for dx, dy in branch.points:
                    px += int(dx * self.scale)
                    py += int(dy * self.scale)
                    points.append((px, py))
                # Orthogonal final connection — snap small gaps to avoid jogs
                dy_gap = abs(py - dst_y)
                if py != dst_y and dy_gap <= 20 and len(points) >= 2:
                    points[-1] = (points[-1][0], dst_y)
                    points.append((dst_x, dst_y))
                elif px != dst_x and py != dst_y:
                    points.append((px, dst_y))
                    points.append((dst_x, dst_y))
                else:
                    points.append((dst_x, dst_y))
                writer.write_polyline(points, CLR_BLACK)
                self._draw_arrow_at_end(writer, points)
            else:
                # L-route: leave horizontal, arrive horizontal
                if start_x == dst_x or start_y == dst_y:
                    points = [(start_x, start_y), (dst_x, dst_y)]
                else:
                    mid_x = (start_x + dst_x) // 2
                    points = [(start_x, start_y), (mid_x, start_y),
                              (mid_x, dst_y), (dst_x, dst_y)]
                writer.write_polyline(points, CLR_BLACK)
                self._draw_arrow_at_end(writer, points)

        # Signal label on branch
        if branch.name and not suppress_wire_labels:
            label_text = branch.name.strip('<>').strip('"')
            if label_text:
                lx = start_x + 5
                ly = start_y - 14
                lw = max(len(label_text) * 7, 20)
                writer.write_text(lx, ly, lw, 12, label_text,
                                  clr=CLR_BLACK, align="horiz. left")


# ---------------------------------------------------------------------------
# PopupGenerator
# ---------------------------------------------------------------------------

class PopupGenerator:
    """Generates popup screens for blocks without auto-generated MEDM screens."""

    def __init__(self, prefix, path, output_dir):
        self.prefix = prefix
        self.path = path
        self.output_dir = output_dir

    def _channel_prefix(self):
        """Build channel prefix like Y1:DMD-PARTICLE."""
        if len(self.prefix) >= 3:
            ifo = self.prefix[:2]
            model = self.prefix[2:]
            path_str = '_'.join(self.path) if self.path else ''
            if path_str:
                return f"{ifo}:{model}-{path_str}"
            else:
                return f"{ifo}:{model}"
        return self.prefix

    def _safe_filename(self, name):
        """Make a safe filename from a block/channel name."""
        return name.replace(':', '_').replace('-', '_').replace(' ', '_')

    def generate_ezca_read_popup(self, block):
        """Generate popup for cdsEzCaRead block. Returns filepath."""
        channel = block.name
        safe = self._safe_filename(channel)
        filename = f"{self.prefix}_EZCAREAD_{safe}.adl"
        filepath = os.path.join(self.output_dir, filename)

        w, h = 400, 150
        writer = AdlWriter()
        writer.write_header(filepath, w, h)

        writer.write_text(5, 5, w - 10, 20, f"EzCaRead: {channel}",
                          clr=CLR_BLACK)
        writer.write_text(5, 35, 80, 20, "Channel:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text(90, 35, w - 95, 20, channel, clr=CLR_BLUE,
                          align="horiz. left")
        writer.write_text(5, 65, 110, 20, "Current Value:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text_update(120, 65, w - 125, 20, channel)
        writer.write_text(5, 95, 80, 20, "Set Value:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text_entry(120, 95, w - 125, 20, channel)

        with open(filepath, 'w') as f:
            f.write(writer.get_text())
        print(f"  Generated popup: {filepath}")
        return filepath

    def generate_ezca_write_popup(self, block):
        """Generate popup for cdsEzCaWrite block. Returns filepath."""
        channel = block.name
        safe = self._safe_filename(channel)
        filename = f"{self.prefix}_EZCAWRITE_{safe}.adl"
        filepath = os.path.join(self.output_dir, filename)

        w, h = 400, 150
        writer = AdlWriter()
        writer.write_header(filepath, w, h)

        writer.write_text(5, 5, w - 10, 20, f"EzCaWrite: {channel}",
                          clr=CLR_BLACK)
        writer.write_text(5, 35, 80, 20, "Channel:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text(90, 35, w - 95, 20, channel, clr=CLR_BLUE,
                          align="horiz. left")
        writer.write_text(5, 65, 110, 20, "Current Value:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text_update(120, 65, w - 125, 20, channel)
        writer.write_text(5, 95, 80, 20, "Set Value:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text_entry(120, 95, w - 125, 20, channel)

        with open(filepath, 'w') as f:
            f.write(writer.get_text())
        print(f"  Generated popup: {filepath}")
        return filepath

    def generate_osc_popup(self, block):
        """Generate popup for cdsOsc block. Returns filepath."""
        chan_prefix = self._channel_prefix() + '_' + block.name
        safe = self._safe_filename(block.name)
        filename = f"{self.prefix}_OSC_{safe}.adl"
        filepath = os.path.join(self.output_dir, filename)

        w, h = 400, 280
        writer = AdlWriter()
        writer.write_header(filepath, w, h)

        path_str = '/'.join(self.path + [block.name])
        writer.write_text(5, 5, w - 10, 20, f"Oscillator: {path_str}",
                          clr=CLR_BLACK)

        y = 35
        for label, suffix, has_entry in [
            ("Frequency:", "_FREQ", True),
            ("Amplitude:", "_AMP", True),
            ("Phase:", "_PHASE", False),
            ("COS out:", "_COS", False),
            ("SIN out:", "_SIN", False),
            ("RAMP out:", "_RAMP", False),
        ]:
            chan = chan_prefix + suffix
            writer.write_text(5, y, 90, 20, label, clr=CLR_BLACK,
                              align="horiz. left")
            writer.write_text_update(100, y, 140, 20, chan)
            if has_entry:
                writer.write_text_entry(250, y, 140, 20, chan)
            y += 35

        with open(filepath, 'w') as f:
            f.write(writer.get_text())
        print(f"  Generated popup: {filepath}")
        return filepath

    def generate_atan2_popup(self, block):
        """Generate popup for cdsAtan2 block. Returns filepath."""
        safe = self._safe_filename(block.name)
        filename = f"{self.prefix}_ATAN2_{safe}.adl"
        filepath = os.path.join(self.output_dir, filename)

        w, h = 300, 150
        writer = AdlWriter()
        writer.write_header(filepath, w, h)

        path_str = '/'.join(self.path + [block.name])
        writer.write_text(5, 5, w - 10, 20, f"Atan2: {path_str}",
                          clr=CLR_BLACK)
        writer.write_text(5, 35, w - 10, 20,
                          "Computes atan2(Y, X)", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text(5, 60, w - 10, 20,
                          f"Inputs: {block.ports_in}  Outputs: {block.ports_out}",
                          clr=CLR_BLACK, align="horiz. left")

        chan_prefix = self._channel_prefix() + '_' + block.name
        writer.write_text(5, 90, 60, 20, "Output:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text_update(70, 90, w - 75, 20, chan_prefix + "_OUT")

        with open(filepath, 'w') as f:
            f.write(writer.get_text())
        print(f"  Generated popup: {filepath}")
        return filepath

    def generate_sqrt_popup(self, block):
        """Generate popup for cdsSqrt block. Returns filepath."""
        safe = self._safe_filename(block.name)
        filename = f"{self.prefix}_SQRT_{safe}.adl"
        filepath = os.path.join(self.output_dir, filename)

        w, h = 300, 150
        writer = AdlWriter()
        writer.write_header(filepath, w, h)

        path_str = '/'.join(self.path + [block.name])
        writer.write_text(5, 5, w - 10, 20, f"Sqrt: {path_str}",
                          clr=CLR_BLACK)
        writer.write_text(5, 35, w - 10, 20,
                          "Computes sqrt(input)", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text(5, 60, w - 10, 20,
                          f"Inputs: {block.ports_in}  Outputs: {block.ports_out}",
                          clr=CLR_BLACK, align="horiz. left")

        chan_prefix = self._channel_prefix() + '_' + block.name
        writer.write_text(5, 90, 60, 20, "Output:", clr=CLR_BLACK,
                          align="horiz. left")
        writer.write_text_update(70, 90, w - 75, 20, chan_prefix + "_OUT")

        with open(filepath, 'w') as f:
            f.write(writer.get_text())
        print(f"  Generated popup: {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate MEDM .adl overview screens from Simulink .mdl files")
    parser.add_argument("mdl_file", help="Path to .mdl file")
    parser.add_argument("--output-dir",
                        help="Output directory for generated .adl files")
    parser.add_argument("--medm-dir",
                        help="Directory containing existing auto-generated .adl files")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for block diagram (default: 1.0)")
    args = parser.parse_args()

    # Read and parse .mdl
    mdl_path = args.mdl_file
    print(f"Parsing {mdl_path}...")
    with open(mdl_path, 'r') as f:
        text = f.read()

    mdl_parser = MdlParser(text)
    system = mdl_parser.parse()

    model_name = mdl_parser.model_name
    ifo = mdl_parser.ifo
    prefix = model_name.upper()

    print(f"Model: {model_name}, IFO: {ifo}, Prefix: {prefix}")
    print(f"Found {len(system.blocks)} top-level blocks, "
          f"{len(system.lines)} top-level lines")
    if system.annotations:
        print(f"Found {len(system.annotations)} annotations")

    # Verify external references found
    ext_refs_found = set()
    _collect_refs(system, model_name, ext_refs_found)
    print(f"Block references found: {len(ext_refs_found)}")

    # Determine output/medm dirs
    if not args.output_dir:
        overview_name = f"{model_name}_overview"
        output_dir = os.path.join("/opt/rtcds/yqg/y1/medm", overview_name)
    else:
        output_dir = args.output_dir

    if not args.medm_dir:
        medm_dir = os.path.join("/opt/rtcds/yqg/y1/medm", model_name)
    else:
        medm_dir = args.medm_dir

    print(f"Output dir: {output_dir}")
    print(f"MEDM dir: {medm_dir}")

    # Generate
    generator = AdlGenerator(model_name, ifo, medm_dir, output_dir,
                             args.scale, host=mdl_parser.host,
                             rate=mdl_parser.rate, dcuid=mdl_parser.dcuid)
    generator.generate_all(system)

    print(f"\nGenerated {len(generator.generated_files)} files total.")


def _collect_refs(system, model_name, refs, path=""):
    """Recursively collect block references for verification."""
    current_path = f"{model_name}/{path}" if not path else path
    for block in system.blocks:
        if block.source_block:
            block_path = '/'.join(
                [p for p in [current_path, block.name] if p])
            refs.add(block_path)
        if block.subsystem:
            sub_path = f"{current_path}/{block.name}" if current_path else block.name
            _collect_refs(block.subsystem, model_name, refs, sub_path)


if __name__ == '__main__':
    main()
