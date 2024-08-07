# GNN for Node Classification on Cora Dataset

This repository contains an implementation of Graph Convolutional Networks (GCNs) using PyTorch Geometric for node classification on the Cora citation network dataset. The objective is to classify academic papers (nodes) into predefined categories based on their content and citation relationships.

## Overview

Node classification is a fundamental problem in graph-based machine learning. In this project, we aim to classify nodes in the Cora dataset, where each node represents an academic paper and edges represent citations between papers. The goal is to assign a category (class) to each paper.

## Problem Description

- **Nodes**: Represent academic papers.
- **Edges**: Represent citation links between papers.
- **Features**: Each node has a feature vector describing the content of the paper.
- **Labels**: Each node is associated with a label indicating the category of the paper.

## Features

- Load and preprocess the Cora dataset
- Define and train a GCN model for node classification
- Visualize training loss and test accuracy
- Inspect and visualize the graph structure

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gnn-node-classification-cora.git
    cd gnn-node-classification-cora
    ```

2. **Create a new Conda environment**:
    ```bash
    conda create -n gnn_env python=3.8
    conda activate gnn_env
    ```

3. **Install PyTorch**:
    ```bash
    # For CPU only:
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    
    # For GPU with CUDA 11.7:
    conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia
    ```

4. **Install PyTorch Geometric and its dependencies**:
    ```bash
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
    ```

## Usage

To run the project, execute the following command:

```bash
python gcn_cora.py
