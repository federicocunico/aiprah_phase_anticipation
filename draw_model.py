# make_model_graph_graphviz.py
import shutil
import sys

try:
    import graphviz
except Exception as e:
    print("❌ Python package 'graphviz' not installed.")
    print("   Fix: pip install graphviz")
    sys.exit(1)

if shutil.which("dot") is None:
    print("❌ Graphviz system executable 'dot' not found.")
    print("   Fix on Ubuntu/Debian:   sudo apt-get update && sudo apt-get install graphviz")
    print("   Fix on macOS (Homebrew): brew install graphviz")
    print("   Fix on Windows (choco):  choco install graphviz")
    sys.exit(1)

from graphviz import Digraph

g = Digraph('DualScaleModel', filename='dual_scale_model', format='svg')
g.attr(rankdir='LR', concentrate='true', fontsize='10')

def box(name, label, color="#dddddd", shape="box", style="rounded,filled"):
    g.node(name, label=label, shape=shape, style=style, fillcolor=color)

# Input
box("inp", "Input\nB×T×3×H×W", "#eeeeee")

# Shared visual encoder
box("bb", "Backbone CNN\n(B·T)×Cb×Hb×Wb", "#cfe8ff")
box("pool", "Spatial Attn / GAP\n(B·T)×Cb", "#cfe8ff")
box("proj", "Proj: Linear+BN+GELU+Drop\nB×T×d", "#cfe8ff")
g.edges([("inp","bb"), ("bb","pool"), ("pool","proj")])

# Long branch
box("slicel", "Slice last T_ℓ\nB×T_ℓ×d", "#d9f7be")
box("mscl", "Multi-Scale TCN\nB×T_ℓ×d", "#d9f7be")
box("confl", "Temporal Conformer\nB×T_ℓ×d", "#d9f7be")
box("pickl", "Pick last\nB×d", "#d9f7be")
g.edge("proj","slicel")
g.edges([("slicel","mscl"), ("mscl","confl"), ("confl","pickl")])

box("phase", "Phase head\nB×C", "#ffd6a5")
box("comp", "Completion head\nB×1", "#ffd6a5")
g.edges([("pickl","phase"), ("pickl","comp")])

box("attnpool", "Attn Pool (learned q)\nB×d", "#ffd6a5")
box("ant", "Anticipation head\nB×C", "#ffd6a5")
g.edge("confl","attnpool")
g.edge("attnpool","ant")

# Short branch
box("slices", "Slice last T_s\nB×T_s×d", "#d9f7be")
box("mscs", "Multi-Scale TCN\nB×T_s×d", "#d9f7be")
box("confs", "Temporal Conformer\nB×T_s×d", "#d9f7be")
box("picks", "Pick last\nB×d", "#d9f7be")
g.edge("proj","slices")
g.edges([("slices","mscs"), ("mscs","confs"), ("confs","picks")])

box("lproj", "Left arm MLP\nB×h_t", "#ffec99")
box("lv", "Left Verb\nB×V", "#ffec99")
box("lt", "Left Target\nB×U", "#ffec99")
g.edges([("picks","lproj"), ("lproj","lv"), ("lproj","lt")])

box("rproj", "Right arm MLP\nB×h_t", "#ffec99")
box("rv", "Right Verb\nB×V", "#ffec99")
box("rt", "Right Target\nB×U", "#ffec99")
g.edges([("picks","rproj"), ("rproj","rv"), ("rproj","rt")])

out = g.render(cleanup=True)
print(f"✅ Wrote {out}")
