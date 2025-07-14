📚 GNN Literature Explorer
GNN Literature Explorer is an interactive Streamlit web application for exploring Graph Neural Networks (GNNs) on academic citation networks such as Cora, Citeseer, and PubMed. It provides dataset visualizations, GNN model training, and graph-based insights — all in one unified interface.

🚀 Features
🧠 GNN Model Training
Train and compare different GNN architectures:

✅ GCN (Graph Convolutional Network)

✅ GAT (Graph Attention Network)

✅ GraphSAGE

Customizable parameters:

🔧 Number of epochs

🔧 Learning rate

🔧 Hidden layer size

📈 Training Feedback

Real-time accuracy plots:

Train Accuracy

Validation Accuracy

Test Accuracy

📊 Dataset Visualization
Explore built-in academic citation datasets:

📚 Cora

📚 Citeseer

📚 PubMed

Visualized information:

📌 Node & edge counts

📌 Feature dimensions per node

📌 Class label distribution

📌 Sample node features

🕸️ Citation subgraph visualization (first 20 nodes)

🌐 Predicted Class Visualization
After model training:

🟡 Visualize the graph with nodes colored by predicted class

🔍 Understand how the trained model interprets structural information

⚠️ Currently only predicted class is shown (no ground-truth comparison)

🔎 Node Search & Neighborhood Exploration
Deep dive into the graph:

🔢 Search a node by its Node ID

🌐 Set neighborhood depth (k-hop) to explore local structure

🎨 Visualize node and its neighbors with predicted class coloring

🛠️ Tech Stack
Python

Streamlit (for UI)

PyTorch Geometric (for GNNs)

NetworkX, Matplotlib, PyVis (for graph visualization)
