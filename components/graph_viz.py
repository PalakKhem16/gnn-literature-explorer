import streamlit as st
from pyvis.network import Network
import networkx as nx
import torch
import tempfile
import os
from torch_geometric.utils import to_networkx

def visualize_graph(data, logits=None):
    try:
        # Convert to NetworkX graph
        G = to_networkx(data, to_undirected=True)
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)

        # Only use first 200 nodes to avoid crashing UI
        limit = 200
        nodes = list(G.nodes)[:limit]
        subgraph = G.subgraph(nodes)

        # Compute predicted classes if logits is provided
        predicted_classes = None
        if logits is not None:
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

        # Color map for up to 10 classes
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

        for node in subgraph.nodes():
            color = "#dddddd"
            label = f"Node {node}"
            if predicted_classes is not None:
                cls = predicted_classes[node]
                color = colors[cls % len(colors)]
                label += f" | Class: {cls}"

            net.add_node(node, label=label, color=color, size=15)

        for source, target in subgraph.edges():
            net.add_edge(source, target)

        net.toggle_physics(True)

        # Save and display
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            path = tmp_file.name
            net.save_graph(path)  # Use save_graph instead of show
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=650, scrolling=True)

        os.unlink(path)

    except Exception as e:
        st.error(f"⚠️ Graph visualization failed: {str(e)}")
