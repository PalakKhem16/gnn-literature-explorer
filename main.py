import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.optim as optim
from torch_geometric.utils import to_networkx

from utils.dataset_loader import load_dataset
from models.gcn import GCN
from models.gat import GAT
from models.graphsage import GraphSAGE
from utils.trainer import train_model
from components.graph_viz import visualize_graph
from components.search_and_expand import search_and_expand


@st.cache_data
def get_graph(_data):
    return to_networkx(_data, to_undirected=True)

def dataset_visualization(dataset, data):
    st.header("📊 Dataset Overview")

    st.markdown(f"""
    - **Nodes**: {data.num_nodes}
    - **Edges**: {data.num_edges}
    - **Features per Node**: {dataset.num_node_features}
    - **Classes**: {dataset.num_classes}
    """)

    st.subheader("🎯 Label Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(data.y.numpy(), bins=dataset.num_classes, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Class Label")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    st.subheader("🧬 Sample Node Features")
    st.dataframe(data.x[:5].numpy())

    st.subheader("🌐 Citation Graph (20-node Subgraph)")
    with st.spinner("Rendering citation graph..."):
        G = get_graph(data)
        sub_nodes = list(G.nodes)[:20]
        sub_G = G.subgraph(sub_nodes)
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        nx.draw_networkx(sub_G, node_size=10, with_labels=False, ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)

def model_training(dataset, data):
    st.header("🧠 Train a GNN Model")

    # Detect dataset change and clear training-related session state
    current_dataset_name = dataset.__class__.__name__ if hasattr(dataset, '__class__') else str(dataset)
    prev_dataset_name = st.session_state.get('prev_dataset_name', None)
    if prev_dataset_name != current_dataset_name:
        for key in ['train_accs', 'val_accs', 'test_accs', 'logits', 'trained']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['prev_dataset_name'] = current_dataset_name

    with st.form("train_form"):
        model_name = st.selectbox("Model Type", ["GCN", "GAT", "GraphSAGE"])
        epochs = st.slider("Epochs", 10, 300, 100)
        lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01)
        hidden_channels = st.number_input("Hidden Channels", min_value=2, max_value=256, value=16)
        submitted = st.form_submit_button("🚀 Start Training")

    # Persist training state and results

    if submitted or st.session_state.get('trained', False):
        if submitted:
            if model_name == "GCN":
                model = GCN(dataset.num_node_features, hidden_channels, dataset.num_classes)
            elif model_name == "GAT":
                model = GAT(dataset.num_node_features, hidden_channels, dataset.num_classes)
            else:
                model = GraphSAGE(dataset.num_node_features, hidden_channels, dataset.num_classes)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            with st.spinner("Training model..."):
                train_accs, val_accs, test_accs = train_model(model, data, optimizer, epochs)

            st.session_state['train_accs'] = train_accs
            st.session_state['val_accs'] = val_accs
            st.session_state['test_accs'] = test_accs

            st.subheader("📈 Accuracy over Epochs")
            fig, ax = plt.subplots()
            ax.plot(train_accs, label='Train')
            ax.plot(val_accs, label='Validation')
            ax.plot(test_accs, label='Test')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig)

            st.success(f"✅ Final Test Accuracy: {test_accs[-1]*100:.2f}%")

            # ...existing code...
            model.eval()
            with st.spinner("Generating predictions..."):
                logits = model(data)
            st.session_state['logits'] = logits
            st.session_state['trained'] = True

        # Use stored results for visualization
        train_accs = st.session_state.get('train_accs')
        val_accs = st.session_state.get('val_accs')
        test_accs = st.session_state.get('test_accs')
        logits = st.session_state.get('logits')

        # ...existing code...

        # Predict and visualize
        st.subheader("🌐 Graph Visualization (Predicted Classes)")
        # Add class color legend
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
        st.markdown("#### Class Color Legend")
        legend_html = ""
        for i, color in enumerate(colors[:dataset.num_classes]):
            legend_html += f"<span style='display:inline-block;width:20px;height:20px;background:{color};margin-right:8px;border-radius:4px;'></span> Class {i} &nbsp;&nbsp;"
        st.markdown(legend_html, unsafe_allow_html=True)

        visualize_graph(data, logits=logits)

    # Node Search/Expansion (only after training)
    if 'logits' in st.session_state:
        st.subheader("🔎 Node Search & Neighborhood Exploration")
        # Add class color legend for node search
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
        st.markdown("#### Class Color Legend")
        legend_html = ""
        for i, color in enumerate(colors[:dataset.num_classes]):
            legend_html += f"<span style='display:inline-block;width:20px;height:20px;background:{color};margin-right:8px;border-radius:4px;'></span> Class {i} &nbsp;&nbsp;"
        st.markdown(legend_html, unsafe_allow_html=True)
        search_and_expand(data, logits=st.session_state['logits'])

def main():
    st.set_page_config(page_title="GNN Literature Explorer", layout="wide")
    st.title("📃 GNN Literature Explorer")

    # Sidebar
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to", [
        "📊 Dataset Visualization",
        "🧠 Model Training"
    ])

    # Dataset
    dataset_name = st.sidebar.selectbox("📂 Choose Dataset", ["Cora", "Citeseer", "PubMed"])
    dataset, data = load_dataset(dataset_name)

    if page == "📊 Dataset Visualization":
        dataset_visualization(dataset, data)
    elif page == "🧠 Model Training":
        model_training(dataset, data)

if __name__ == "__main__":
    main()
