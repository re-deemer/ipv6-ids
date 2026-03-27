# ipv6-ids
IPv6 Security

# IPv6 Neighbor Discovery / Router Advertisement — ML-based IDS

> **Master's Thesis Project**  
> Machine Learning Intrusion Detection System for IPv6 ND/RA Attacks  
> Built on a Containerlab Digital Twin Network Environment

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Network Topology — Digital Twin](#2-network-topology--digital-twin)
3. [Attack Simulations](#3-attack-simulations)
4. [Traffic Capture](#4-traffic-capture)
5. [Prerequisites & Installation](#5-prerequisites--installation)
6. [Project Structure](#6-project-structure)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Pipeline Architecture](#8-pipeline-architecture)
9. [Feature Engineering](#9-feature-engineering)
10. [Machine Learning Methodology](#10-machine-learning-methodology)
11. [Output Files](#11-output-files)
12. [Visualisations](#12-visualisations)
13. [Configuration Reference](#13-configuration-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Background & Theory](#15-background--theory)

---

## 1. Project Overview

This project implements a **machine learning–based Intrusion Detection System (IDS)** designed to detect and classify malicious **IPv6 Neighbor Discovery (ND)** and **Router Advertisement (RA)** attacks in real time. The IDS is trained and validated against traffic captured from an **18-node Containerlab digital twin** of a real IPv6 network.

The pipeline ingests a raw packet capture file (`raw_capture.pcap`), extracts 30 per-packet features from the IPv6 and ICMPv6 layers, selects the most discriminative features using **SelectKBest with χ² scoring**, and trains a **Random Forest classifier** to distinguish three traffic classes:

| Class Label | ICMPv6 Type | Description |
|---|---|---|
| `Normal` | 133, 136, other | Legitimate IPv6 / ICMPv6 traffic |
| `RA_Attack` | 134 | Router Advertisement flood (`atk6-flood_router26`) |
| `ND_Attack` | 135 | Neighbor Solicitation exhaustion (`atk6-flood_solicitate6`) |

Key evaluation metrics reported are **Balanced Accuracy**, **Macro F1-Score**, and **standard Accuracy** — evaluated across a sweep of feature counts k ∈ {5, 10, 15, 20, 25, 30} and validated with **10-fold stratified cross-validation**.

---

## 2. Network Topology — Digital Twin

The experimental environment is a **Containerlab** topology named `ipv6-research` consisting of 18 nodes.

```
┌─────────────────────────────────────────────────────────┐
│                  Containerlab: ipv6-research             │
│                                                         │
│   [router]──────┐                                       │
│   (radvd/RA)    │                                       │
│                 ▼                                       │
│   [attacker]──► [switch-br] ◄──── [victim × 15]        │
│   (Kali/THC)    (br0 bridge)       (tcpdump eth1)       │
└─────────────────────────────────────────────────────────┘
```

| Node | Container Name | Role |
|---|---|---|
| `switch-br` | `clab-ipv6-research-switch-br` | Linux bridge container (alpine:latest) — internal Layer 2 switch connecting all nodes via an internal `br0` bridge; keeps the host machine clean |
| `router` | `clab-ipv6-research-router` | Legitimate Router Advertisement source running **radvd** (alpine:latest) |
| `victim` × 15 | `clab-ipv6-research-victim` | Traffic sensor nodes running **tcpdump** on `eth1` — primary capture point |
| `attacker` | `clab-ipv6-research-attacker` | Kali Linux node running the **THC-IPv6** toolkit |

The switch-br node hosts an internal Linux bridge (`br0`) inside its container. All multicast attack traffic from the attacker propagates through this bridge and reaches all victim nodes on the same Layer 2 segment.

---

## 3. Attack Simulations

Two distinct THC-IPv6 attacks were executed from the `attacker` node against the live topology:

### 3.1 RA Flood Attack — `atk6-flood_router26`

```bash
# Executed on the attacker node
atk6-flood_router26 eth1
```

**Mechanism:** Floods the network with a high-rate burst of **fake Router Advertisement packets** (ICMPv6 Type 134). Each fake RA carries a randomised source link-local address (`fe80::/10`) and either a zero or maximum (`65535`) router lifetime, causing victim nodes to repeatedly update and corrupt their default gateway routing tables.

**Network fingerprint:**
- ICMPv6 Type = 134
- `hop_limit` = 255 (RFC-mandated)
- Source always in `fe80::/10`, destination `ff02::1` (all-nodes multicast)
- Inter-arrival time < 0.1 ms (burst pattern)
- Packet size 86–120 bytes

### 3.2 ND Exhaustion Attack — `atk6-flood_solicitate6`

```bash
# Executed on the attacker node
atk6-flood_solicitate6 eth1
```

**Mechanism:** Floods the network with **Neighbor Solicitation packets** (ICMPv6 Type 135) targeting randomised IPv6 addresses. Victim nodes are forced to allocate neighbor cache entries for each solicited address, exhausting kernel memory and degrading or crashing the neighbor discovery subsystem.

**Network fingerprint:**
- ICMPv6 Type = 135
- `hop_limit` = 255
- Randomised target addresses (not unspecified `::`)
- Inter-arrival time < 0.05 ms (very bursty)
- Packet size 72–96 bytes

---

## 4. Traffic Capture

**tcpdump** was run simultaneously on:
- All 15 victim nodes (`eth1` interface)
- The attacker machine (for ground-truth labelling)

```bash
# Command run on each victim node
tcpdump -i eth1 -w raw_capture.pcap
```

The resulting `raw_capture.pcap` contains a mix of normal IPv6 traffic (Router Solicitations, Neighbor Advertisements, etc.) interleaved with both attack types. This file is the primary input to the ML pipeline.

---

## 5. Prerequisites & Installation

### Python Version

Python 3.8 or higher is required.

### Required Libraries

All libraries listed below must be installed. If running in VS Code with a dedicated virtual environment, install with:

```bash
pip install numpy pandas matplotlib scikit-learn plotly scapy
```

| Library | Purpose |
|---|---|
| `numpy` | Numerical array operations |
| `pandas` | DataFrame construction and CSV I/O |
| `matplotlib` | Static plots (curves, bar charts, confusion matrix) |
| `scikit-learn` | ML pipeline — RandomForest, SelectKBest, cross-validation, metrics |
| `plotly` | Interactive 3-D classification scatter (HTML output) |
| `scapy` | Raw pcap parsing and IPv6 / ICMPv6 layer dissection |

> **Note:** `scapy` is only required when running with a real pcap file. If using the `--synthetic` flag, scapy is not imported and does not need to be installed.

### VS Code Setup

1. Open the project folder in VS Code.
2. Select your Python interpreter (bottom-left status bar → `Python x.x.x`).
3. Open the integrated terminal (`Ctrl+`` ` or `View → Terminal`).
4. Install dependencies if not already present (see above).
5. Place `raw_capture.pcap` in the same folder as `ipv6_ids_pipeline.py`.

---

## 6. Project Structure

```
project-root/
│
├── ipv6_ids_pipeline.py          ← Main pipeline script
├── raw_capture.pcap              ← Raw packet capture (place here before running)
│
├── README.md                     ← This file
│
│   ── Generated outputs (created on first run) ──
│
├── ipv6_ids_dataset.csv          ← Extracted & labelled feature dataset
├── feature_selection_curve.png   ← Feature count vs accuracy + CV band plot
├── confusion_matrix_best_k.png   ← Confusion matrix at optimal k
├── comparative_line_graph.png    ← Rolling-mean time-series per class
├── comparative_bar_graph.png     ← Class distribution + feature mean bars
├── per_class_metrics.png         ← Precision / Recall / F1 grouped bar chart
└── interactive_3d_classification.html  ← Interactive Plotly 3-D scatter
```

---

## 7. Running the Pipeline

### Standard run (real pcap)

Place `raw_capture.pcap` in the same directory as the script, then run:

```bash
python ipv6_ids_pipeline.py
```

The script will automatically detect the pcap file, extract features, and proceed through the full ML pipeline.

### Synthetic demo (no pcap required)

```bash
python ipv6_ids_pipeline.py --synthetic
```

This generates a statistically realistic synthetic dataset (12,000 samples: 40% Normal, 35% RA_Attack, 25% ND_Attack) and runs the full pipeline without needing Scapy or a pcap file. This mode is useful for testing the pipeline on a new machine or demonstrating it without access to the original capture.

### Automatic fallback

If `raw_capture.pcap` is not found and `--synthetic` was not passed, the script automatically falls back to synthetic mode and prints a warning:

```
[!] pcap not found at './raw_capture.pcap'. Switching to synthetic dataset.
```

---

## 8. Pipeline Architecture

The pipeline executes the following 12 steps sequentially:

```
Step 1  │ Load pcap OR generate synthetic dataset
        ↓
Step 2  │ Print dataset shape and class distribution
        ↓
Step 3  │ Save labelled DataFrame → ipv6_ids_dataset.csv
        ↓
Step 4  │ Preprocess: LabelEncoder + MinMaxScaler → X, y arrays
        ↓
Step 5  │ Validate k sweep values against available feature count
        ↓
Step 6  │ For each k ∈ {5, 10, 15, 20, 25, 30}:
        │   ├─ SelectKBest (χ²) on training set
        │   ├─ 10-fold Stratified CV → CV Balanced Acc, Macro F1
        │   └─ Final fit + evaluate on held-out test set
        ↓
Step 7  │ Print per-k feature ranking table (top-5 per k)
        ↓
Step 8  │ Print dedicated accuracy report for top-5 features
        ↓
Step 9  │ Print full classification report at optimal k
        ↓
Step 10 │ Print full results summary table
        ↓
Step 11 │ Print optimal k recommendation with selected features
        ↓
Step 12 │ Generate all 6 output plots
```

---

## 9. Feature Engineering

The pipeline extracts **30 per-packet features** grouped into four categories. All features are normalised to [0, 1] with MinMaxScaler before feature selection.

### Layer-3 IPv6 Features (10 features)

| Feature | Description |
|---|---|
| `ipv6_version` | IP version field (always 6) |
| `ipv6_traffic_class` | DSCP / ECN traffic class byte |
| `ipv6_flow_label` | 20-bit flow label field |
| `ipv6_payload_length` | Payload length in bytes |
| `ipv6_next_header` | Next header protocol number (58 = ICMPv6) |
| `ipv6_hop_limit` | Hop limit (TTL equivalent); attacks always use 255 |
| `ipv6_src_is_link_local` | 1 if source address is in `fe80::/10` |
| `ipv6_dst_is_multicast` | 1 if destination is multicast (`ff00::/8`) |
| `ipv6_src_prefix_16` | First 16-bit group of source address (integer) |
| `ipv6_dst_prefix_16` | First 16-bit group of destination address (integer) |

### ICMPv6 Presence & Type Features (8 features)

| Feature | Description |
|---|---|
| `has_icmpv6` | 1 if packet contains any ICMPv6 layer |
| `has_ra` | 1 if ICMPv6 Router Advertisement (Type 134) present |
| `has_ns` | 1 if ICMPv6 Neighbor Solicitation (Type 135) present |
| `has_na` | 1 if ICMPv6 Neighbor Advertisement (Type 136) present |
| `has_rs` | 1 if ICMPv6 Router Solicitation (Type 133) present |
| `icmpv6_type` | Raw ICMPv6 type number |
| `icmpv6_code` | Raw ICMPv6 code number |
| `icmpv6_payload_len` | Byte length of the ICMPv6 payload |

### RA-Specific Features — ICMPv6 Type 134 (6 features)

These features are only populated for Router Advertisement packets; all others are zero-padded.

| Feature | Description |
|---|---|
| `ra_cur_hop_limit` | Advertised current hop limit field |
| `ra_flags` | Combined M (managed) + O (other config) flag bits |
| `ra_router_lifetime` | Router lifetime in seconds (0 or 65535 = suspicious) |
| `ra_reachable_time` | Reachable time field in milliseconds |
| `ra_retrans_timer` | Retransmission timer in milliseconds |
| `ra_num_options` | Number of options appended to the RA message |

### NS-Specific Features — ICMPv6 Type 135 (2 features)

| Feature | Description |
|---|---|
| `ns_target_is_unspecified` | 1 if target address is `::` (the unspecified address) |
| `ns_has_src_lladdr_option` | 1 if Src Link-Layer Address option is present |

### Temporal / Flow-Level Features (3 features)

| Feature | Description |
|---|---|
| `pkt_size` | Total packet size in bytes (L2 frame) |
| `inter_arrival_ms` | Time since the previous IPv6 packet in milliseconds |
| `cumulative_pkt_count` | Sequential packet index within the capture |

---

## 10. Machine Learning Methodology

### 10.1 Train/Test Split

The dataset is divided using an **80/20 stratified split** (stratified on the `label` column) with `random_state=42` to ensure reproducibility. Stratification guarantees that the class proportions are preserved in both the training and test sets.

### 10.2 Feature Selection — SelectKBest with χ²

`SelectKBest` with the **chi-squared (χ²) statistic** is applied exclusively to the training portion to prevent data leakage. The χ² test measures the statistical independence between each feature and the target class label — a higher score means greater discriminative power.

The feature selector is swept across k ∈ {5, 10, 15, 20, 25, 30}, producing a learning curve showing how classification performance changes as the feature budget increases.

### 10.3 Random Forest Classifier

A **Random Forest** with the following configuration is trained for each value of k:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Sufficient for stable variance reduction |
| `class_weight` | `"balanced"` | Compensates for class imbalance in real captures |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallelises tree building across all CPU cores |

### 10.4 Cross-Validation

**10-fold Stratified K-Fold cross-validation** is applied to the training set for each k. This produces mean and standard deviation estimates of:
- Balanced Accuracy
- Macro F1-Score
- Standard Accuracy

Cross-validation is performed on the training fold only; the held-out test set is never touched during this step.

### 10.5 Evaluation Metrics

Three metrics are reported for every k:

| Metric | Formula / Definition | Why It Matters Here |
|---|---|---|
| **Balanced Accuracy** | Mean of per-class recall | Robust to class imbalance — attack classes may be minority classes in a real capture |
| **Macro F1-Score** | Unweighted mean of per-class F1 | Penalises poor performance on any single class equally |
| **Accuracy** | Correct predictions / total | Standard metric; can be misleading with imbalanced classes |

---

## 11. Output Files

After a successful run, the following files are written to the working directory:

| File | Type | Description |
|---|---|---|
| `ipv6_ids_dataset.csv` | CSV | Full labelled feature dataset extracted from the pcap |
| `feature_selection_curve.png` | PNG | Dual-panel: test metrics vs k (left) + 10-fold CV with ±1 std band (right) |
| `confusion_matrix_best_k.png` | PNG | Confusion matrix at the optimal k (highest Balanced Accuracy) |
| `comparative_line_graph.png` | PNG | Rolling-mean time-series of top-5 features, one line per traffic class |
| `comparative_bar_graph.png` | PNG | Class count distribution + mean ± std of top-5 features grouped by class |
| `per_class_metrics.png` | PNG | Grouped bar chart: Precision, Recall, F1 per class at optimal k |
| `interactive_3d_classification.html` | HTML | Plotly interactive 3-D scatter — open in any browser |

---

## 12. Visualisations

### 12.1 Feature Selection Curve (`feature_selection_curve.png`)

A dual-panel chart showing how classification performance changes as the number of selected features k increases.

- **Left panel:** Three lines on the held-out test set — Test Accuracy (blue), Test Balanced Accuracy (orange), Test Macro F1 (green). A vertical dashed line marks the optimal k.
- **Right panel:** 10-fold CV Balanced Accuracy and Macro F1 on the training set, shown as lines with a ±1 standard deviation shaded band.

This plot is the primary tool for selecting the final feature budget for deployment.

### 12.2 Confusion Matrix (`confusion_matrix_best_k.png`)

A heatmap confusion matrix evaluated on the held-out test set at the optimal k. Rows represent the true class, columns represent the predicted class. Perfect detection appears as a diagonal matrix.

### 12.3 Comparative Line Graph (`comparative_line_graph.png`)

For each of the top-5 most important features (by χ² score), a sub-plot shows the **rolling mean** (window = 100 packets) of that feature's value over packet index, with a separate coloured line per class:

- **Blue** — Normal traffic
- **Orange/Red** — RA_Attack
- **Green** — ND_Attack

This visualisation highlights how attack traffic is temporally distinct from normal traffic in key feature dimensions such as `inter_arrival_ms` and `ipv6_hop_limit`.

### 12.4 Comparative Bar Graph (`comparative_bar_graph.png`)

A multi-panel bar chart with one panel per top-5 feature plus a class distribution overview:

- **Leftmost panel:** Total packet counts per class (Normal / RA_Attack / ND_Attack).
- **Remaining panels:** Mean ± standard deviation of each feature grouped by class, with exact mean values annotated above each bar.

### 12.5 Per-Class Metrics Chart (`per_class_metrics.png`)

A grouped bar chart showing **Precision**, **Recall**, and **F1-Score** side by side for each of the three traffic classes at the optimal k. Exact values are annotated above each bar for thesis reporting.

### 12.6 Interactive 3-D Scatter (`interactive_3d_classification.html`)

A fully interactive Plotly scatter plot projecting the test-set samples into the 3-D space defined by the three highest-ranked features. Open this file in any modern browser (Chrome, Firefox, Edge).

- **Colour** encodes the true class label.
- **Circle markers** (●) indicate correct predictions.
- **X markers** (✕) indicate misclassifications.
- **Hover tooltip** shows true label, predicted label, and the exact feature values for any individual point.
- The plot can be rotated, zoomed, and panned interactively. Traces can be toggled in the legend.

---

## 13. Configuration Reference

The following constants at the top of `ipv6_ids_pipeline.py` can be adjusted without changing the pipeline logic:

| Constant | Default | Description |
|---|---|---|
| `PCAP_FILE` | `"./raw_capture.pcap"` | Path to the input pcap file |
| `CSV_OUTPUT` | `"ipv6_ids_dataset.csv"` | Output path for the labelled CSV dataset |
| `K_SWEEP` | `[5, 10, 15, 20, 25, 30]` | Feature count values to evaluate |
| `RANDOM_STATE` | `42` | Global random seed for reproducibility |
| `N_FOLDS` | `10` | Number of cross-validation folds |
| `TEST_SIZE` | `0.20` | Proportion of data held out for testing |
| `N_TREES` | `100` | Number of trees in the Random Forest |

---

## 14. Troubleshooting

**`ImportError: No module named 'scapy'`**  
Scapy is not installed. Either install it (`pip install scapy`) or run with the `--synthetic` flag to bypass pcap parsing entirely.

**`MemoryError` during Random Forest training**  
Reduce `N_TREES` (e.g., to 50) or reduce the synthetic dataset size in `generate_synthetic_dataset(n_samples=...)`. On Windows, also ensure you are running inside `if __name__ == "__main__":` to avoid multiprocessing issues — this is already handled in the script.

**`FileNotFoundError: raw_capture.pcap`**  
The script will automatically fall back to synthetic mode. If you intend to use a real capture, verify the path set in `PCAP_FILE` matches the actual location of the file.

**Plotly 3-D graph does not open**  
The file `interactive_3d_classification.html` is saved to disk. Open it manually by double-clicking it in your file explorer, or drag it into a browser window. It does not open automatically from the terminal.

**Plots are blank or not saved (headless server)**  
The script uses `matplotlib.use("Agg")` to force the non-interactive Agg backend, which writes files without opening a display window. This is correct behaviour on servers and in VS Code integrated terminals. The PNG files are saved to the working directory.

**`ValueError: k` exceeds number of features**  
This occurs if the pcap produces fewer than 30 IPv6 packets. The script automatically filters `K_SWEEP` to only include values ≤ the actual feature count and adjusts gracefully.

---

## 15. Background & Theory

### IPv6 Neighbor Discovery Protocol (NDP)

NDP (RFC 4861) is the IPv6 replacement for ARP. It uses five ICMPv6 message types to handle address resolution, router discovery, and duplicate address detection on a link. The messages most relevant to this project are:

- **Router Advertisement (Type 134):** Sent by routers to announce their presence and network configuration parameters (prefix, MTU, hop limit) to hosts.
- **Neighbor Solicitation (Type 135):** Sent by a node to discover the link-layer address of a neighbour or to verify that a cached address is still reachable.

### Attack Threat Model

**RA Flood (atk6-flood_router26):** Because NDP has no built-in authentication (absent SEND / RFC 3971), any node on the link can forge Router Advertisement packets. An attacker can inject hundreds of fake RAs per second, each advertising itself as the default router. Victim hosts update their routing table with each valid-looking RA, causing routing instability, denial-of-service, or traffic redirection (man-in-the-middle).

**ND Exhaustion (atk6-flood_solicitate6):** Each Neighbor Solicitation for a previously unseen address forces the receiving host to allocate a `INCOMPLETE` entry in its neighbor cache. By flooding with solicitations targeting random addresses, the attacker exhausts the fixed-size neighbor cache, causing legitimate address resolution to fail and potentially crashing the network stack.

### Why Random Forest?

Random Forests are well-suited to this problem for several reasons: they handle mixed numerical features without strong distributional assumptions, they are robust to the high variance of network traffic features, they provide feature importances natively (complementing SelectKBest), and they tolerate class imbalance well when `class_weight="balanced"` is set.

### Why χ² Feature Selection?

After MinMax scaling to [0, 1], all features are non-negative, satisfying the χ² test's requirement. The χ² statistic directly measures the statistical dependence between each feature and the class label, ranking features by how much information they individually contribute to classification. This is computationally inexpensive and produces an interpretable ranking that directly informs which network-layer fields are the strongest attack indicators.

---

*Master's Thesis — IPv6 Network Security | Containerlab Digital Twin | ML-based IDS*
