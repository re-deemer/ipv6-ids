"""
=============================================================================
 IPv6 Neighbor Discovery / Router Advertisement  –  ML-based IDS
 Master's Thesis  |  Containerlab Digital Twin  |  Random Forest + SelectKBest
=============================================================================

USAGE
-----
  # Real capture (requires scapy + raw_capture.pcap in the working directory)
  python ipv6_ids_pipeline.py

  # Force synthetic demo (no pcap needed)
  python ipv6_ids_pipeline.py --synthetic

PIPELINE OVERVIEW
-----------------
  1. Feature extraction from pcap  (or synthetic generation)
  2. Save DataFrame → ipv6_ids_dataset.csv
  3. SelectKBest (chi2) at k ∈ {5, 10, 15, 20, 25, 30}
  4. 80/20 stratified train/test split  +  10-fold stratified CV on train set
  5. Random Forest classifier
  6. Metrics: Balanced Accuracy, Macro-F1, Accuracy
  7. Feature-count vs Test-Accuracy plot  →  feature_selection_curve.png
  8. Comparative Line, Bar, and Interactive 3D Plotly graphs

CLASSES
-------
  0 = Normal    (all non-attack ICMPv6 / IPv6 traffic)
  1 = RA_Attack (ICMPv6 Type 134 flood  –  atk6-flood_router26)
  2 = ND_Attack (ICMPv6 Type 135 flood  –  atk6-flood_solicitate6)
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import os
import time
import warnings
import textwrap
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – safe on servers / containers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS / CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PCAP_FILE       = "./raw_capture.pcap"
CSV_OUTPUT      = "ipv6_ids_dataset.csv"
PLOT_CURVE      = "feature_selection_curve.png"
PLOT_CM         = "confusion_matrix_best_k.png"
PLOT_LINE       = "comparative_line_graph.png"
PLOT_BAR        = "comparative_bar_graph.png"
PLOT_3D         = "interactive_3d_classification.html"

# ICMPv6 type → label mapping
ICMPV6_TYPE_RA  = 134    # Router Advertisement
ICMPV6_TYPE_NS  = 135    # Neighbor Solicitation  (ND attack)
ICMPV6_TYPE_NA  = 136    # Neighbor Advertisement
ICMPV6_TYPE_RS  = 133    # Router Solicitation

LABEL_NORMAL    = "Normal"
LABEL_RA        = "RA_Attack"
LABEL_ND        = "ND_Attack"
CLASS_NAMES     = [LABEL_NORMAL, LABEL_RA, LABEL_ND]

# Colours consistent across all plots
CLASS_COLORS    = {
    LABEL_NORMAL : "#2196F3",   # blue
    LABEL_RA     : "#FF5722",   # deep-orange
    LABEL_ND     : "#4CAF50",   # green
}

# k values to sweep
K_SWEEP         = [5, 10, 15, 20, 25, 30]

RANDOM_STATE    = 42
N_FOLDS         = 10
TEST_SIZE       = 0.20
N_TREES         = 100           # Random Forest trees


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – PCAP FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_from_pcap(pcap_path: str) -> pd.DataFrame:
    """
    Parse raw_capture.pcap with Scapy and extract per-packet features.

    Feature groups
    ──────────────
    Layer-3 (IPv6)
      ipv6_version, ipv6_traffic_class, ipv6_flow_label,
      ipv6_payload_length, ipv6_next_header, ipv6_hop_limit,
      ipv6_src_is_link_local, ipv6_dst_is_multicast,
      ipv6_src_prefix_16, ipv6_dst_prefix_16

    ICMPv6
      icmpv6_type, icmpv6_code, icmpv6_payload_len,
      has_icmpv6, has_ra, has_ns, has_na, has_rs

    RA-specific (Type 134)
      ra_cur_hop_limit, ra_flags, ra_router_lifetime,
      ra_reachable_time, ra_retrans_timer, ra_num_options

    NS-specific (Type 135)
      ns_target_is_unspecified, ns_has_src_lladdr_option

    Temporal / flow-level
      pkt_size, inter_arrival_time (ms), cumulative_pkt_count

    Label
      label  → 'Normal' | 'RA_Attack' | 'ND_Attack'
    """
    try:
        from scapy.all import rdpcap, IPv6, ICMPv6ND_RA, ICMPv6ND_NS
        from scapy.layers.inet6 import (
            ICMPv6ND_NA, ICMPv6ND_RS,
            ICMPv6NDOptPrefixInfo, ICMPv6NDOptSrcLLAddr,
        )
    except ImportError:
        raise ImportError(
            "Scapy is not installed. Run:  pip install scapy\n"
            "Or use --synthetic flag to run with generated data."
        )

    print(f"[+] Reading pcap: {pcap_path}")
    pkts = rdpcap(pcap_path)
    print(f"    Total packets loaded: {len(pkts):,}")

    records  = []
    prev_time = None

    def _ipv6_prefix16(addr: str) -> int:
        """Return the first 16-bit group of an IPv6 address as an integer."""
        if not addr:
            return 0
        try:
            if '::' in addr:
                left, _, right = addr.partition('::')
                left_groups  = left.split(':')  if left  else []
                right_groups = right.split(':') if right else []
                missing      = 8 - len(left_groups) - len(right_groups)
                groups       = left_groups + ['0'] * missing + right_groups
            else:
                groups = addr.split(':')
            return int(groups[0] or '0', 16)
        except (ValueError, IndexError):
            return 0

    for idx, pkt in enumerate(pkts):
        rec = {}

        if IPv6 not in pkt:
            continue

        ip6 = pkt[IPv6]
        src = str(ip6.src)
        dst = str(ip6.dst)

        rec["ipv6_version"]           = int(ip6.version)
        rec["ipv6_traffic_class"]     = int(ip6.tc)
        rec["ipv6_flow_label"]        = int(ip6.fl)
        rec["ipv6_payload_length"]    = int(ip6.plen)
        rec["ipv6_next_header"]       = int(ip6.nh)
        rec["ipv6_hop_limit"]         = int(ip6.hlim)
        rec["ipv6_src_is_link_local"] = int(src.startswith("fe80"))
        rec["ipv6_dst_is_multicast"]  = int(dst.startswith("ff"))
        rec["ipv6_src_prefix_16"]     = _ipv6_prefix16(src)
        rec["ipv6_dst_prefix_16"]     = _ipv6_prefix16(dst)

        has_ra = int(ICMPv6ND_RA in pkt)
        has_ns = int(ICMPv6ND_NS in pkt)
        has_na = int(ICMPv6ND_NA in pkt)
        has_rs = int(ICMPv6ND_RS in pkt)
        has_icmpv6 = int(has_ra | has_ns | has_na | has_rs)

        icmpv6_type = 0
        icmpv6_code = 0
        for layer in pkt.layers():
            ln = layer.__name__
            if "ICMPv6" in ln:
                try:
                    icmpv6_type = int(pkt[layer].type)
                    icmpv6_code = int(pkt[layer].code)
                except Exception:
                    pass
                break

        rec["has_icmpv6"]         = has_icmpv6
        rec["has_ra"]             = has_ra
        rec["has_ns"]             = has_ns
        rec["has_na"]             = has_na
        rec["has_rs"]             = has_rs
        rec["icmpv6_type"]        = icmpv6_type
        rec["icmpv6_code"]        = icmpv6_code
        rec["icmpv6_payload_len"] = len(bytes(pkt[IPv6].payload)) if has_icmpv6 else 0

        if has_ra:
            ra = pkt[ICMPv6ND_RA]
            rec["ra_cur_hop_limit"]   = int(ra.chlim)
            rec["ra_flags"]           = int(ra.M) * 2 + int(ra.O)
            rec["ra_router_lifetime"] = int(ra.routerlifetime)
            rec["ra_reachable_time"]  = int(ra.reachabletime)
            rec["ra_retrans_timer"]   = int(ra.retranstimer)
            num_opts = 0
            layer = ra.payload
            while layer:
                num_opts += 1
                layer = layer.payload if hasattr(layer, "payload") else None
            rec["ra_num_options"] = num_opts
        else:
            for f in ["ra_cur_hop_limit", "ra_flags", "ra_router_lifetime",
                      "ra_reachable_time", "ra_retrans_timer", "ra_num_options"]:
                rec[f] = 0

        if has_ns:
            ns  = pkt[ICMPv6ND_NS]
            tgt = str(ns.tgt) if hasattr(ns, "tgt") else ""
            rec["ns_target_is_unspecified"] = int(tgt == "::")
            rec["ns_has_src_lladdr_option"] = int(ICMPv6NDOptSrcLLAddr in pkt)
        else:
            rec["ns_target_is_unspecified"] = 0
            rec["ns_has_src_lladdr_option"] = 0

        rec["pkt_size"] = len(pkt)
        pkt_time = float(pkt.time)
        if prev_time is None:
            rec["inter_arrival_ms"] = 0.0
        else:
            rec["inter_arrival_ms"] = max(0.0, (pkt_time - prev_time) * 1000.0)
        prev_time = pkt_time
        rec["cumulative_pkt_count"] = idx + 1

        if icmpv6_type == ICMPV6_TYPE_RA:
            rec["label"] = LABEL_RA
        elif icmpv6_type == ICMPV6_TYPE_NS:
            rec["label"] = LABEL_ND
        else:
            rec["label"] = LABEL_NORMAL

        records.append(rec)

    df = pd.DataFrame(records)
    print(f"    IPv6 packets extracted: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – SYNTHETIC DATASET GENERATOR  (demo / unit-test fallback)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(n_samples: int = 12_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Produce a labelled dataset that mirrors the statistical fingerprints of
    real RA-flood and ND-exhaustion captures from THC-IPv6.

    Class split  (roughly realistic for a mixed-capture session):
      Normal    → 40 %
      RA_Attack → 35 %
      ND_Attack → 25 %
    """
    rng = np.random.default_rng(seed)

    n_normal = int(n_samples * 0.40)
    n_ra     = int(n_samples * 0.35)
    n_nd     = n_samples - n_normal - n_ra

    def _make_normal(n):
        d = {}
        d["ipv6_version"]           = np.full(n, 6)
        d["ipv6_traffic_class"]     = rng.integers(0, 2,  size=n)
        d["ipv6_flow_label"]        = rng.integers(0, 100_000, size=n)
        d["ipv6_payload_length"]    = rng.integers(24, 1500, size=n)
        d["ipv6_next_header"]       = rng.choice([58, 59, 17, 6], size=n, p=[0.5,0.1,0.2,0.2])
        d["ipv6_hop_limit"]         = rng.integers(50, 255, size=n)
        d["ipv6_src_is_link_local"] = rng.integers(0, 2, size=n)
        d["ipv6_dst_is_multicast"]  = rng.integers(0, 2, size=n)
        d["ipv6_src_prefix_16"]     = rng.integers(0x2001, 0xFE80, size=n)
        d["ipv6_dst_prefix_16"]     = rng.integers(0x2001, 0xFE80, size=n)
        d["has_icmpv6"]             = rng.integers(0, 2, size=n)
        d["has_ra"]                 = np.zeros(n, int)
        d["has_ns"]                 = np.zeros(n, int)
        d["has_na"]                 = rng.integers(0, 2, size=n)
        d["has_rs"]                 = rng.integers(0, 2, size=n)
        d["icmpv6_type"]            = rng.choice([136, 133, 0], size=n, p=[0.4,0.2,0.4])
        d["icmpv6_code"]            = np.zeros(n, int)
        d["icmpv6_payload_len"]     = rng.integers(0, 100, size=n)
        for f in ["ra_cur_hop_limit","ra_flags","ra_router_lifetime",
                  "ra_reachable_time","ra_retrans_timer","ra_num_options"]:
            d[f] = np.zeros(n, int)
        d["ns_target_is_unspecified"] = np.zeros(n, int)
        d["ns_has_src_lladdr_option"] = np.zeros(n, int)
        d["pkt_size"]               = rng.integers(60, 1500, size=n)
        d["inter_arrival_ms"]       = rng.exponential(5.0, size=n)
        d["cumulative_pkt_count"]   = np.arange(1, n+1)
        d["label"]                  = np.full(n, LABEL_NORMAL)
        return pd.DataFrame(d)

    def _make_ra_attack(n):
        d = {}
        d["ipv6_version"]           = np.full(n, 6)
        d["ipv6_traffic_class"]     = np.zeros(n, int)
        d["ipv6_flow_label"]        = rng.integers(0, 10, size=n)
        d["ipv6_payload_length"]    = rng.integers(56, 80, size=n)
        d["ipv6_next_header"]       = np.full(n, 58)
        d["ipv6_hop_limit"]         = np.full(n, 255)
        d["ipv6_src_is_link_local"] = np.ones(n, int)
        d["ipv6_dst_is_multicast"]  = np.ones(n, int)
        d["ipv6_src_prefix_16"]     = np.full(n, 0xFE80)
        d["ipv6_dst_prefix_16"]     = np.full(n, 0xFF02)
        d["has_icmpv6"]             = np.ones(n, int)
        d["has_ra"]                 = np.ones(n, int)
        d["has_ns"]                 = np.zeros(n, int)
        d["has_na"]                 = np.zeros(n, int)
        d["has_rs"]                 = np.zeros(n, int)
        d["icmpv6_type"]            = np.full(n, 134)
        d["icmpv6_code"]            = np.zeros(n, int)
        d["icmpv6_payload_len"]     = rng.integers(56, 80, size=n)
        d["ra_cur_hop_limit"]       = rng.choice([0, 64], size=n)
        d["ra_flags"]               = rng.integers(0, 4, size=n)
        d["ra_router_lifetime"]     = rng.choice([0, 65535], size=n)
        d["ra_reachable_time"]      = rng.integers(0, 3600000, size=n)
        d["ra_retrans_timer"]       = rng.integers(0, 100000, size=n)
        d["ra_num_options"]         = rng.integers(1, 4, size=n)
        d["ns_target_is_unspecified"] = np.zeros(n, int)
        d["ns_has_src_lladdr_option"] = np.zeros(n, int)
        d["pkt_size"]               = rng.integers(86, 120, size=n)
        d["inter_arrival_ms"]       = rng.exponential(0.05, size=n)
        d["cumulative_pkt_count"]   = np.arange(1, n+1)
        d["label"]                  = np.full(n, LABEL_RA)
        return pd.DataFrame(d)

    def _make_nd_attack(n):
        d = {}
        d["ipv6_version"]           = np.full(n, 6)
        d["ipv6_traffic_class"]     = np.zeros(n, int)
        d["ipv6_flow_label"]        = rng.integers(0, 10, size=n)
        d["ipv6_payload_length"]    = rng.integers(32, 48, size=n)
        d["ipv6_next_header"]       = np.full(n, 58)
        d["ipv6_hop_limit"]         = np.full(n, 255)
        d["ipv6_src_is_link_local"] = np.ones(n, int)
        d["ipv6_dst_is_multicast"]  = np.ones(n, int)
        d["ipv6_src_prefix_16"]     = np.full(n, 0xFE80)
        d["ipv6_dst_prefix_16"]     = np.full(n, 0xFF02)
        d["has_icmpv6"]             = np.ones(n, int)
        d["has_ra"]                 = np.zeros(n, int)
        d["has_ns"]                 = np.ones(n, int)
        d["has_na"]                 = np.zeros(n, int)
        d["has_rs"]                 = np.zeros(n, int)
        d["icmpv6_type"]            = np.full(n, 135)
        d["icmpv6_code"]            = np.zeros(n, int)
        d["icmpv6_payload_len"]     = rng.integers(24, 40, size=n)
        for f in ["ra_cur_hop_limit","ra_flags","ra_router_lifetime",
                  "ra_reachable_time","ra_retrans_timer","ra_num_options"]:
            d[f] = np.zeros(n, int)
        d["ns_target_is_unspecified"] = rng.integers(0, 2, size=n)
        d["ns_has_src_lladdr_option"] = rng.integers(0, 2, size=n)
        d["pkt_size"]               = rng.integers(72, 96, size=n)
        d["inter_arrival_ms"]       = rng.exponential(0.03, size=n)
        d["cumulative_pkt_count"]   = np.arange(1, n+1)
        d["label"]                  = np.full(n, LABEL_ND)
        return pd.DataFrame(d)

    df = pd.concat([_make_normal(n_normal),
                    _make_ra_attack(n_ra),
                    _make_nd_attack(n_nd)],
                   ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """Encode labels, MinMax-scale features, return X / y / names / encoder."""
    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)

    feature_cols = [c for c in df.columns if c != "label"]
    X_raw        = df[feature_cols].fillna(0).values.astype(np.float64)

    scaler = MinMaxScaler()
    X      = scaler.fit_transform(X_raw)

    return X, y, feature_cols, le


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – SELECTKBEST SWEEP  +  RANDOM FOREST  +  10-FOLD CV
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(X, y, feature_names, k_values=K_SWEEP):
    """
    For each k in k_values:
      1. SelectKBest(chi2, k) on TRAINING set
      2. 10-fold Stratified CV on training set
      3. Final fit → evaluate on held-out test set
    Returns results DataFrame, X_test (full features), y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n{'─'*60}")
    print(f"  Train samples : {len(X_train):,}   |   Test samples : {len(X_test):,}")
    print(f"  Total features: {X.shape[1]}")
    print(f"{'─'*60}")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results        = []
    k_values_valid = [k for k in k_values if k <= X.shape[1]]

    for k in k_values_valid:
        t0 = time.time()

        selector     = SelectKBest(chi2, k=k)
        X_train_sel  = selector.fit_transform(X_train, y_train)
        X_test_sel   = selector.transform(X_test)
        selected_idx = selector.get_support(indices=True)
        sel_names    = [feature_names[i] for i in selected_idx]
        scores       = selector.scores_[selected_idx]
        top_features = sorted(zip(sel_names, scores),
                              key=lambda x: x[1], reverse=True)

        clf = RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        cv_scores = cross_validate(
            clf, X_train_sel, y_train,
            cv=cv,
            scoring={
                "balanced_acc": "balanced_accuracy",
                "macro_f1":     "f1_macro",
                "accuracy":     "accuracy",
            },
            return_train_score=False,
        )

        clf.fit(X_train_sel, y_train)
        y_pred        = clf.predict(X_test_sel)

        test_acc      = accuracy_score(y_test, y_pred)
        test_bal_acc  = balanced_accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        elapsed       = time.time() - t0

        results.append({
            "k":               k,
            "top_features":    top_features,
            "selected_names":  sel_names,
            "cv_acc_mean":     cv_scores["test_accuracy"].mean(),
            "cv_acc_std":      cv_scores["test_accuracy"].std(),
            "cv_bal_acc_mean": cv_scores["test_balanced_acc"].mean(),
            "cv_bal_acc_std":  cv_scores["test_balanced_acc"].std(),
            "cv_f1_mean":      cv_scores["test_macro_f1"].mean(),
            "cv_f1_std":       cv_scores["test_macro_f1"].std(),
            "test_accuracy":   test_acc,
            "test_balanced_acc": test_bal_acc,
            "test_f1_macro":   test_f1_macro,
            "_clf":            clf,
            "_selector":       selector,
            "_y_pred":         y_pred,
            "_elapsed":        elapsed,
        })

        print(
            f"  k={k:>3d}  |  "
            f"CV Bal-Acc={cv_scores['test_balanced_acc'].mean():.4f}±"
            f"{cv_scores['test_balanced_acc'].std():.4f}  |  "
            f"CV F1={cv_scores['test_macro_f1'].mean():.4f}  |  "
            f"Test Acc={test_acc:.4f}  |  "
            f"Test Bal-Acc={test_bal_acc:.4f}  |  "
            f"Test F1={test_f1_macro:.4f}  |  "
            f"[{elapsed:.1f}s]"
        )

    return pd.DataFrame(results), X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_table(results_df: pd.DataFrame):
    print(f"\n{'═'*60}")
    print("  FEATURE RANKING PER k  (top-5 shown per run)")
    print(f"{'═'*60}")
    for _, row in results_df.iterrows():
        k = row["k"]
        print(f"\n  ── k = {k} ──────────────────────────────────────")
        print(f"  {'Feature':<35}  {'χ² Score':>12}")
        print(f"  {'─'*35}  {'─'*12}")
        for feat, score in row["top_features"][:5]:
            print(f"  {feat:<35}  {score:>12.2f}")


def print_top5_accuracy(results_df: pd.DataFrame):
    """Report accuracy metrics when only the 5 most important features are used."""
    row = results_df[results_df["k"] == 5]
    if row.empty:
        row = results_df.iloc[[0]]

    row = row.iloc[0]
    print(f"\n{'═'*60}")
    print("  RESULTS — TOP-5 MOST IMPORTANT FEATURES")
    print(f"{'═'*60}")
    print(f"  {'Feature':<35}  {'χ² Score':>12}")
    print(f"  {'─'*35}  {'─'*12}")
    for feat, score in row["top_features"][:5]:
        print(f"  {feat:<35}  {score:>12.2f}")
    print(f"\n  Test Accuracy      : {row['test_accuracy']:.4f}")
    print(f"  Test Balanced Acc  : {row['test_balanced_acc']:.4f}")
    print(f"  Test Macro F1      : {row['test_f1_macro']:.4f}")
    print(f"  CV  Balanced Acc   : {row['cv_bal_acc_mean']:.4f} ± {row['cv_bal_acc_std']:.4f}")
    print(f"  CV  Macro F1       : {row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}")


def print_classification_report_best(results_df, y_test, le):
    best_row = results_df.loc[results_df["test_balanced_acc"].idxmax()]
    k_best   = best_row["k"]
    y_pred   = best_row["_y_pred"]
    print(f"\n{'═'*60}")
    print(f"  DETAILED CLASSIFICATION REPORT  (k = {k_best}  — Best Balanced Acc)")
    print(f"{'═'*60}")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                zero_division=0))
    return best_row


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – PLOTS  (matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

METRIC_COLORS = {
    "test_accuracy":     "#2196F3",
    "test_balanced_acc": "#FF5722",
    "test_f1_macro":     "#4CAF50",
}


def plot_feature_selection_curve(results_df: pd.DataFrame, out_path: str):
    """Feature count vs test metric  +  CV ± std band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(
        "Random Forest — Feature Count vs. Performance\n"
        "(SelectKBest χ²  |  80/20 split  |  10-fold CV)",
        fontsize=13, fontweight="bold"
    )
    k_vals = results_df["k"].values

    ax1 = axes[0]
    ax1.plot(k_vals, results_df["test_accuracy"],
             "o-", color=METRIC_COLORS["test_accuracy"],
             lw=2, ms=7, label="Test Accuracy")
    ax1.plot(k_vals, results_df["test_balanced_acc"],
             "s-", color=METRIC_COLORS["test_balanced_acc"],
             lw=2, ms=7, label="Test Balanced Accuracy")
    ax1.plot(k_vals, results_df["test_f1_macro"],
             "^-", color=METRIC_COLORS["test_f1_macro"],
             lw=2, ms=7, label="Test Macro F1")

    best_idx = results_df["test_balanced_acc"].idxmax()
    best_k   = results_df.loc[best_idx, "k"]
    best_val = results_df.loc[best_idx, "test_balanced_acc"]
    ax1.axvline(best_k, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax1.annotate(
        f"Optimal k={best_k}\n({best_val:.4f})",
        xy=(best_k, best_val),
        xytext=(best_k + 0.8, best_val - 0.04),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="grey"),
        color="grey",
    )
    ax1.set_xlabel("Number of Features (k)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Hold-out Test Set Metrics", fontsize=11)
    ax1.set_xticks(k_vals)
    low = max(0, results_df[["test_accuracy","test_balanced_acc","test_f1_macro"]].min().min() - 0.05)
    ax1.set_ylim(low, 1.02)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.35)
    ax1.spines[["top","right"]].set_visible(False)

    ax2 = axes[1]
    for mean_col, std_col, color, label in [
        ("cv_bal_acc_mean", "cv_bal_acc_std", METRIC_COLORS["test_balanced_acc"], "CV Balanced Acc"),
        ("cv_f1_mean",      "cv_f1_std",      METRIC_COLORS["test_f1_macro"],     "CV Macro F1"),
    ]:
        means = results_df[mean_col].values
        stds  = results_df[std_col].values
        ax2.plot(k_vals, means, "o-", color=color, lw=2, ms=6, label=label)
        ax2.fill_between(k_vals, means - stds, means + stds, alpha=0.18, color=color)

    ax2.set_xlabel("Number of Features (k)", fontsize=11)
    ax2.set_ylabel("Score (mean ± 1 std)", fontsize=11)
    ax2.set_title(f"{N_FOLDS}-Fold CV on Training Set", fontsize=11)
    ax2.set_xticks(k_vals)
    low2 = max(0, results_df[["cv_bal_acc_mean","cv_f1_mean"]].min().min() - 0.05)
    ax2.set_ylim(low2, 1.02)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.35)
    ax2.spines[["top","right"]].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[+] Feature selection curve saved → {out_path}")


def plot_confusion_matrix(best_row, y_test, le, out_path: str):
    y_pred = best_row["_y_pred"]
    k      = best_row["k"]
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix  (k={k}, Best Balanced Accuracy)\n"
        f"Test Bal-Acc = {best_row['test_balanced_acc']:.4f}   "
        f"Macro-F1 = {best_row['test_f1_macro']:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[+] Confusion matrix saved  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – COMPARATIVE LINE GRAPH
#  Shows per-class mean values for the top-5 selected features across packets.
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparative_line(df: pd.DataFrame, best_row, out_path: str):
    """
    For each of the top-5 selected features, plot the smoothed rolling mean
    of that feature over packet index, one line per class.
    This highlights how attack traffic differs from normal traffic over time.
    """
    top5_names = [feat for feat, _ in best_row["top_features"][:5]]

    fig, axes = plt.subplots(
        len(top5_names), 1,
        figsize=(14, 4 * len(top5_names)),
        constrained_layout=True,
        sharex=False,
    )
    fig.suptitle(
        "Comparative Line Graph — Top-5 Features per Traffic Class\n"
        "(Rolling mean over sorted packet index, window = 100)",
        fontsize=13, fontweight="bold",
    )

    if len(top5_names) == 1:
        axes = [axes]

    window = 100

    for ax, feat in zip(axes, top5_names):
        if feat not in df.columns:
            ax.set_title(f"{feat}  [not in dataframe]")
            continue

        for cls in CLASS_NAMES:
            subset = df[df["label"] == cls][feat].reset_index(drop=True).astype(float)
            if len(subset) < window:
                smoothed = subset
            else:
                smoothed = subset.rolling(window=window, min_periods=1).mean()
            ax.plot(
                smoothed.index,
                smoothed.values,
                lw=1.6,
                alpha=0.85,
                color=CLASS_COLORS[cls],
                label=cls,
            )

        ax.set_ylabel(feat, fontsize=9)
        ax.set_xlabel("Packet index (within class)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[+] Comparative line graph saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – COMPARATIVE BAR GRAPH
#  Two sub-plots: class distribution  +  mean feature values per class.
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparative_bar(df: pd.DataFrame, best_row, out_path: str):
    """
    Left panel  : Class distribution (packet counts)
    Right panels: Mean value of each top-5 feature grouped by class
    """
    top5_names = [feat for feat, _ in best_row["top_features"][:5]]

    n_panels = 1 + len(top5_names)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4 * n_panels, 5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Comparative Bar Graph — Class Distribution & Feature Means\n"
        "(Top-5 SelectKBest χ² features)",
        fontsize=13, fontweight="bold",
    )

    # ── Left panel: class counts ──────────────────────────────────────────────
    ax0    = axes[0]
    counts = [df[df["label"] == c].shape[0] for c in CLASS_NAMES]
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    bars   = ax0.bar(CLASS_NAMES, counts, color=colors, edgecolor="white",
                     linewidth=1.2, width=0.55)
    for bar, cnt in zip(bars, counts):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.01,
                 f"{cnt:,}", ha="center", va="bottom", fontsize=9)
    ax0.set_title("Class Distribution", fontsize=10)
    ax0.set_ylabel("Packet Count", fontsize=9)
    ax0.set_ylim(0, max(counts) * 1.15)
    ax0.tick_params(axis="x", labelsize=8)
    ax0.spines[["top","right"]].set_visible(False)
    ax0.grid(axis="y", alpha=0.3)

    # ── Feature panels ────────────────────────────────────────────────────────
    for ax, feat in zip(axes[1:], top5_names):
        if feat not in df.columns:
            ax.set_title(f"{feat}\n[missing]", fontsize=9)
            continue

        means  = [df[df["label"] == c][feat].mean() for c in CLASS_NAMES]
        stds   = [df[df["label"] == c][feat].std()  for c in CLASS_NAMES]
        bars   = ax.bar(CLASS_NAMES, means, yerr=stds,
                        color=colors, edgecolor="white",
                        linewidth=1.2, width=0.55,
                        error_kw=dict(ecolor="black", capsize=4, elinewidth=1))
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(means) * 0.02 if max(means) > 0 else 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("Mean ± Std", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[+] Comparative bar graph saved  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – INTERACTIVE 3-D PLOTLY GRAPH
#  Projects the test-set predictions into the 3-D space of the 3 most
#  discriminative features (by χ² score).  Colour = true label,
#  marker symbol = predicted label (correct vs. wrong).
# ══════════════════════════════════════════════════════════════════════════════

def plot_interactive_3d(df: pd.DataFrame, best_row, le, out_path: str):
    """
    Interactive 3-D scatter (Plotly HTML) showing the three most important
    features on X / Y / Z axes.  Points are coloured by TRUE class and
    shaped by PREDICTED class, making misclassifications immediately visible.
    """
    top3 = [feat for feat, _ in best_row["top_features"][:3]]
    # Pad to 3 if fewer features were selected
    all_feat = [feat for feat, _ in best_row["top_features"]]
    while len(top3) < 3 and len(all_feat) >= 3:
        top3 = all_feat[:3]
    if len(top3) < 3:
        print("[!] Fewer than 3 features selected; skipping 3-D plot.")
        return

    feat_x, feat_y, feat_z = top3

    # ── Reconstruct test-set samples with labels ──────────────────────────────
    # We'll use the full df with a random-state-matched split so indices align
    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )
    test_df = test_df.copy().reset_index(drop=True)

    y_pred_labels = le.inverse_transform(best_row["_y_pred"])
    test_df["predicted_label"] = y_pred_labels
    test_df["correct"]         = (test_df["label"] == test_df["predicted_label"])

    # Sample for performance (max 3000 points)
    if len(test_df) > 3000:
        test_df = test_df.sample(3000, random_state=RANDOM_STATE).reset_index(drop=True)

    # Plotly colour map
    color_map = {
        LABEL_NORMAL: "#2196F3",
        LABEL_RA:     "#FF5722",
        LABEL_ND:     "#4CAF50",
    }

    symbol_map = {True: "circle", False: "x"}   # correct / wrong

    fig = go.Figure()

    for cls in CLASS_NAMES:
        for correct, sym_name in [(True, "circle"), (False, "x")]:
            mask    = (test_df["label"] == cls) & (test_df["correct"] == correct)
            sub     = test_df[mask]
            if sub.empty:
                continue
            opacity = 0.75 if correct else 1.0
            size    = 4    if correct else 7

            fig.add_trace(go.Scatter3d(
                x=sub[feat_x],
                y=sub[feat_y],
                z=sub[feat_z],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color_map[cls],
                    symbol=sym_name,
                    opacity=opacity,
                    line=dict(width=0.5, color="white") if correct else dict(width=1, color="black"),
                ),
                name=f"{cls} — {'✓ Correct' if correct else '✗ Wrong'}",
                text=sub.apply(
                    lambda r: (
                        f"True: {r['label']}<br>"
                        f"Pred: {r['predicted_label']}<br>"
                        f"{feat_x}: {r[feat_x]:.4f}<br>"
                        f"{feat_y}: {r[feat_y]:.4f}<br>"
                        f"{feat_z}: {r[feat_z]:.4f}"
                    ), axis=1
                ),
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(
            text=(
                f"Interactive 3D Classification View  (k={best_row['k']})<br>"
                f"<sub>Axes: {feat_x} | {feat_y} | {feat_z}  —  "
                f"Circles = correct predictions, X = misclassifications</sub>"
            ),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title=feat_x,
            yaxis_title=feat_y,
            zaxis_title=feat_z,
            xaxis=dict(backgroundcolor="rgb(240,240,255)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(240,255,240)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(255,240,240)", gridcolor="white"),
        ),
        legend=dict(
            title="<b>Class — Prediction</b>",
            x=0.0, y=1.0,
            font=dict(size=10),
        ),
        width=1050,
        height=750,
        margin=dict(l=0, r=0, b=30, t=80),
        paper_bgcolor="white",
    )

    fig.write_html(out_path)
    print(f"[+] Interactive 3-D graph saved → {out_path}  (open in browser)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – BONUS: Per-class metric bar chart (precision / recall / F1)
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_class_metrics(best_row, y_test, le, out_path: str = "per_class_metrics.png"):
    """
    Grouped bar chart: precision, recall, F1 for each class at best k.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_pred    = best_row["_y_pred"]
    k         = best_row["k"]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(le.classes_)), zero_division=0
    )

    x    = np.arange(len(le.classes_))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    b1 = ax.bar(x - w,  prec, w, label="Precision", color="#2196F3", edgecolor="white")
    b2 = ax.bar(x,      rec,  w, label="Recall",    color="#FF5722", edgecolor="white")
    b3 = ax.bar(x + w,  f1,   w, label="F1-Score",  color="#4CAF50", edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(le.classes_, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Per-Class Metrics — Precision / Recall / F1  (k={k})\n"
        f"Balanced Acc={best_row['test_balanced_acc']:.4f}   Macro F1={best_row['test_f1_macro']:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[+] Per-class metrics chart saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════╗
    ║  IPv6 ND / RA Attack Detection  –  ML IDS Pipeline       ║
    ║  Random Forest + SelectKBest  |  Containerlab Twin       ║
    ║  Master's Thesis Project                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """))


def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    USE_SYNTHETIC = "--synthetic" in sys.argv   # pass --synthetic to skip pcap
    PCAP_PATH     = PCAP_FILE
    # ─────────────────────────────────────────────────────────────────────────

    print_banner()

    # ── Step 1: Load / Generate dataset ──────────────────────────────────────
    use_synthetic = USE_SYNTHETIC or not os.path.isfile(PCAP_PATH)

    if use_synthetic:
        if not USE_SYNTHETIC:
            print(f"[!] pcap not found at '{PCAP_PATH}'. Switching to synthetic dataset.")
        else:
            print("[*] --synthetic flag set. Using generated dataset.")
        df = generate_synthetic_dataset()
    else:
        print(f"[+] Found pcap at '{PCAP_PATH}'. Extracting features …")
        df = extract_features_from_pcap(PCAP_PATH)

    # ── Step 2: Shape & class distribution ───────────────────────────────────
    print(f"\n{'═'*60}")
    print("  DATASET OVERVIEW")
    print(f"{'═'*60}")
    print(f"  Shape          : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"  Feature columns: {df.shape[1] - 1}")
    print(f"  Label column   : 'label'")
    print("\n  Class Distribution:")
    vc = df["label"].value_counts()
    for cls in CLASS_NAMES:
        count = vc.get(cls, 0)
        pct   = 100 * count / len(df)
        bar   = "█" * int(pct / 2)
        print(f"    {cls:<12}  {count:>6,}  ({pct:5.1f}%)  {bar}")

    # ── Step 3: Save CSV ──────────────────────────────────────────────────────
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\n[+] Dataset saved → {CSV_OUTPUT}")

    # ── Step 4: Preprocess ────────────────────────────────────────────────────
    X, y, feature_names, le = preprocess(df)
    print(f"\n[+] Encoded classes : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Step 5: Validate k values ─────────────────────────────────────────────
    k_vals = sorted({v for v in K_SWEEP if v <= len(feature_names)})
    if not k_vals:
        k_vals = [len(feature_names)]
    print(f"[+] k sweep         : {k_vals}")

    # ── Step 6: Run pipeline ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RUNNING SelectKBest + Random Forest  (10-Fold CV)")
    print(f"{'='*60}")
    results_df, X_test_full, y_test_full = run_pipeline(X, y, feature_names, k_vals)

    # ── Step 7: Feature ranking table ─────────────────────────────────────────
    print_feature_table(results_df)

    # ── Step 8: Top-5 feature accuracy report ─────────────────────────────────
    print_top5_accuracy(results_df)

    # ── Step 9: Classification report for best k ──────────────────────────────
    best_row = print_classification_report_best(results_df, y_test_full, le)

    # ── Step 10: Summary table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY TABLE")
    print(f"{'='*60}")
    summary = results_df[[
        "k",
        "cv_bal_acc_mean","cv_bal_acc_std",
        "cv_f1_mean","cv_f1_std",
        "test_accuracy","test_balanced_acc","test_f1_macro",
    ]].copy()
    summary.columns = [
        "k",
        "CV_BalAcc_mean","CV_BalAcc_std",
        "CV_F1_mean","CV_F1_std",
        "Test_Acc","Test_BalAcc","Test_F1",
    ]
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Step 11: Optimal k ────────────────────────────────────────────────────
    best_k   = best_row["k"]
    best_bal = best_row["test_balanced_acc"]
    best_f1  = best_row["test_f1_macro"]
    print(f"\n{'='*60}")
    print(f"  ★  OPTIMAL k = {best_k}")
    print(f"     Test Balanced Acc = {best_bal:.4f}")
    print(f"     Test Macro F1     = {best_f1:.4f}")
    print(f"  Selected Features:")
    for fn, sc in best_row["top_features"]:
        print(f"    - {fn:<38}  chi2={sc:.2f}")
    print(f"{'='*60}")

    # ── Step 12: All plots ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  GENERATING PLOTS …")
    print(f"{'='*60}")

    # 12a. Feature selection curve (matplotlib)
    plot_feature_selection_curve(results_df, PLOT_CURVE)

    # 12b. Confusion matrix (matplotlib)
    plot_confusion_matrix(best_row, y_test_full, le, PLOT_CM)

    # 12c. Comparative line graph (matplotlib)
    plot_comparative_line(df, best_row, PLOT_LINE)

    # 12d. Comparative bar graph (matplotlib)
    plot_comparative_bar(df, best_row, PLOT_BAR)

    # 12e. Per-class precision/recall/F1 bar chart (matplotlib)
    plot_per_class_metrics(best_row, y_test_full, le)

    # 12f. Interactive 3-D Plotly scatter (HTML – open in browser)
    plot_interactive_3d(df, best_row, le, PLOT_3D)

    print(f"\n{'='*60}")
    print("  OUTPUT FILES")
    print(f"{'='*60}")
    outputs = [
        (CSV_OUTPUT,           "Labelled dataset (CSV)"),
        (PLOT_CURVE,           "Feature selection curve (PNG)"),
        (PLOT_CM,              "Confusion matrix (PNG)"),
        (PLOT_LINE,            "Comparative line graph (PNG)"),
        (PLOT_BAR,             "Comparative bar graph (PNG)"),
        ("per_class_metrics.png", "Per-class Precision/Recall/F1 (PNG)"),
        (PLOT_3D,              "Interactive 3-D classification (HTML — open in browser)"),
    ]
    for fname, desc in outputs:
        if os.path.isfile(fname):
            size_kb = os.path.getsize(fname) / 1024
            print(f"  ✓  {fname:<40}  {desc}  [{size_kb:.0f} KB]")
        else:
            print(f"  ✗  {fname:<40}  (not generated)")

    print("\n[OK] Pipeline complete.\n")
    return results_df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = main()