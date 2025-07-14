import streamlit as st
from pyvis.network import Network
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
import tempfile
import os

def search_and_expand(data, logits=None):
    st.subheader("🔍 Search & Explore Node Neighborhood")

    if data is None or data.num_nodes == 0:
        st.warning("Graph not loaded yet.")
        return

    G = to_networkx(data, to_undirected=True)

    predicted_classes = None
    if logits is not None:
        try:
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
        except Exception:
            st.warning("Could not parse model predictions.")

    # Reliable default to prevent errors
    node_id = st.number_input("Enter Node ID", min_value=0, max_value=max(1, data.num_nodes - 1), value=0, step=1)
    hop = st.slider("Neighborhood Depth (hop)", min_value=1, max_value=min(10, data.num_nodes - 1), value=1)

    try:
        neighbors = nx.single_source_shortest_path_length(G, node_id, cutoff=hop)
        sub_nodes = list(neighbors.keys())
        subgraph = G.subgraph(sub_nodes)

        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

        for node in subgraph.nodes():
            color = "#cccccc"
            title = f"Node {node}"
            if predicted_classes is not None:
                cls = predicted_classes[node]
                color = colors[cls % len(colors)]
                title += f"<br>Predicted Class: {cls}"
            size = 25 if node == node_id else 15
            net.add_node(node, label=str(node), color=color, size=size, title=title)

        for src, tgt in subgraph.edges():
            net.add_edge(src, tgt)

        net.toggle_physics(True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=650, scrolling=True)
        os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Error visualizing node: {e}")
