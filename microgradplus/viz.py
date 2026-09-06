"""
viz.py computation graph visualization, using graphviz.

This is optional (requires the "graphviz" Python package and the
graphviz system binary). It's useful for sanity-checking small graphs by
eye exactly as in the original notebook
"""


def trace(root):
    """Return (nodes, edges) reachable from root via _prev"""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, rankdir="LR"):
    """Render the computation graph rooted at `root` as a graphviz Digraph

    Each Value is drawn as a record node showing its label, data, and
    grad; each operation is drawn as a separate small node feeding into
    its output
    """
    from graphviz import Digraph

    dot = Digraph(format="svg", graph_attr={"rankdir": rankdir})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
