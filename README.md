🧠 DDoS Attack Detection in AMI Networks using Graph Convolutional Networks (GCN)
📘 Overview

This project implements a Graph Convolutional Network (GCN) to detect DDoS (Distributed Denial of Service) attacks in Advanced Metering Infrastructure (AMI) networks.
It uses supervised learning on labelled network traffic data to classify flows as normal or attack.

The GCN model leverages the relational structure in network traffic, representing each flow as a graph node with associated features (IP, ports, protocol, timestamps, etc.).

⚙️ Key Features

End-to-end GCN-based intrusion detection pipeline

Converts network traffic data into a graph structure for training

Implements feature scaling, encoding, and preprocessing

Evaluates multiple metrics such as Accuracy, Precision, Recall, Specificity, AUC

Saves the trained model using Joblib for later inference


🧠 Model Architecture

The GCN model consists of:

Input Layer: Number of input features (after preprocessing)

Hidden Layer: 32 hidden units with ReLU activation

Output Layer: 2 output nodes (Normal / Attack)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

🧩 Dataset Description

The dataset combined_dataset_40.csv contains network traffic records with features like:

Source IP, Destination IP, Source Port, Destination Port

Protocol, Timestamp

Target (0 = Normal, 1 = Attack)

Each IP is converted into 4 numeric features for graph compatibility, and all values are scaled using StandardScaler.

🚀 Training and Evaluation

The dataset is split into 80% training and 20% testing.

The model is trained for 200 epochs using the Adam optimizer.

After training, the model’s state is saved as gcn_model.joblib.



📊 Results

Typical results achieved on the dataset:

Metric	Value (Example)
Accuracy	0.98
Precision	0.97
Recall (Sensitivity)	0.98
Specificity	0.96
F1 Score	0.975
AUC	0.985

A Receiver Operating Characteristic (ROC) curve is also plotted to visualise model performance.

🧩 Technologies Used

Python 3.10+

PyTorch & PyTorch Geometric

scikit-learn

pandas, numpy, matplotlib


👩‍💻 Author

Nithyasree Jagatheesan
