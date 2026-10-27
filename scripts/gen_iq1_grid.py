#!/usr/bin/env python3
"""Emit ds4.c's CPU-side IQ1 grid from the authoritative ggml-common.h table.

The grid is 2048 uint64 entries, each packing 8 signed bytes drawn from
{0xff = -1, 0x00 = 0, 0x01 = +1}.  Hand-copying 16 KiB of hex would fail
silently in a way that shows up only as subtly wrong logits, so extract it and
assert every invariant worth asserting: the count, the byte alphabet, and the
two endpoints that are easy to eyeball in the source.

Usage: gen_iq1_grid.py <ggml-common.h> <out.inc>
"""
import re
import sys

BEGIN = re.compile(r'GGML_TABLE_BEGIN\(\s*uint64_t\s*,\s*iq1s_grid\s*,\s*NGRID_IQ1S\s*\)')
HEX = re.compile(r'0x([0-9a-fA-F]{16})')


def main(src, dst):
    text = open(src, encoding='utf-8').read()

    m = BEGIN.search(text)
    if not m:
        print("error: uint64_t iq1s_grid table not found", file=sys.stderr)
        return 1
    end = text.find('GGML_TABLE_END()', m.end())
    if end < 0:
        print("error: unterminated table", file=sys.stderr)
        return 1

    vals = [int(h, 16) for h in HEX.findall(text[m.end():end])]

    if len(vals) != 2048:
        print(f"error: got {len(vals)} entries, expected 2048", file=sys.stderr)
        return 1
    for v in vals:
        for j in range(8):
            b = (v >> (8 * j)) & 0xFF
            if b not in (0x00, 0x01, 0xFF):
                print(f"error: entry 0x{v:016x} has byte 0x{b:02x} outside "
                      f"the {{-1,0,+1}} alphabet", file=sys.stderr)
                return 1
    if vals[0] != 0xffffffffffffffff or vals[-1] != 0x0101010101010101:
        print("error: endpoints do not match the source table", file=sys.stderr)
        return 1

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(
            "/* Generated from cuda/mmq/ggml-common.h by\n"
            " * scripts/gen_iq1_grid.py -- do not edit by hand.\n"
            " *\n"
            " * The IQ1_S/IQ1_M codebook: 2048 entries, each packing 8 signed\n"
            " * bytes from {0xff = -1, 0x00 = 0, 0x01 = +1}.  ggml keeps this\n"
            " * table behind GGML_COMMON_IMPL_C, which ds4.c cannot define\n"
            " * without pulling in the rest of a C++ header, so it is mirrored\n"
            " * here for the CPU reference path.\n"
            " */\n"
            f"static const uint64_t iq1s_grid[{len(vals)}] = {{\n")
        for i in range(0, len(vals), 4):
            row = ', '.join(f"0x{v:016x}" for v in vals[i:i + 4])
            f.write(f"    {row},\n")
        f.write("};\n")

    print(f"wrote {dst}: {len(vals)} entries, all bytes in {{-1,0,+1}}, "
          f"endpoints match")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
