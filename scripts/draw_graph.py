# RENDERS THE COMPILED LANGGRAPH TOPOLOGY TO A DIAGRAM.
import sys
from pathlib import Path

# MAKE THE PROJECT ROOT IMPORTABLE
sys.path.insert(0, str(Path(__file__).parent.parent))

from newsroomagent.graph import build_graph

graph = build_graph([])
g = graph.get_graph()

out_dir = Path(__file__).parent.parent / "docs"
out_dir.mkdir(exist_ok=True)

# CREATE MERMAID TEXT
mermaid = g.draw_mermaid()
(out_dir / "architecture_graph.mmd").write_text(mermaid, encoding="utf-8")
print(mermaid)

# CREATE IMAGE
try:
    (out_dir / "architecture_graph.png").write_bytes(g.draw_mermaid_png())
    print(f"IMAGE CREATED")
except Exception as e:
    print(f"PNG SKIPPED")
