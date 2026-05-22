#!/usr/bin/env python3
# =============================================================================
# ML-Based IDS for IPv6 ND & RA Attack Detection — v18 EXTERNAL-VALIDATION
#
# ── CHANGES FROM v17 STRATIFIED ─────────────────────────────────────────────
#
#  MODE-1  Three operating modes via --mode CLI flag:
#            • train          : extract markers + features, train, save bundle
#            • evaluate-pcap  : load bundle, score an external PCAP file
#            • evaluate-csv   : load bundle, score an external feature CSV
#  MODE-2  Single-file inputs only: ONE PCAP or ONE CSV per invocation.
#  MODE-3  In-band marker parser replaces events.csv / nodes.csv loaders.
#            Phase timeline + node identity reconstructed from ICMPv6 echo
#            packets with source 2001:db8::ffff carrying magic payload.
#            *** v4 CSV FALLBACK: when no markers found in PCAP, the pipeline
#            automatically looks for raw_capture_events.csv and
#            raw_capture_nodes.csv beside the PCAP and loads those instead.
#            Override with --events-csv and --nodes-csv flags. ***
#  MODE-4  Joblib model bundle: scaler + label encoder + feature columns +
#            top-k selection + both trained classifiers (XGB + RF) +
#            hyperparameters, persisted to a single file.
#  MODE-5  External CSV loader with column renaming and label-mapping JSON.
#  MODE-6  External evaluation function with per-class metrics, ROC/PR-AUC,
#            and append-only results table for the thesis.
#
#  All v17 STRATIFIED science preserved:
#  Stratified 80/20 Split A (random_state=42) + Split B (random_state=123)
#  5-Fold StratifiedKFold cross-validation
#  SMOTE oversampling on training data only (when needed)
#  Top-10 feature selection (ANOVA-F, train-only)
#  XGBoost + RF hyperparameter sweeps with StratifiedKFold
#  Single-feature leakage scan
#  Bootstrap 95% CI for per-class and macro-F1
#  DummyClassifier baseline + learning curves
#  Multi-model comparison across Split A vs Split B
#  5 visualisation outputs (PNG)
# =============================================================================

import warnings; warnings.filterwarnings("ignore")
import os;       os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import argparse, math, sys, json, csv as _csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import joblib

from scapy.all import (PcapReader, IPv6, ICMPv6ND_RA, ICMPv6ND_NS, ICMPv6ND_NA,
                       Ether, ICMPv6EchoRequest, ICMPv6EchoReply)

from sklearn.dummy            import DummyClassifier
from sklearn.base             import BaseEstimator, ClassifierMixin
from sklearn.ensemble         import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection  import (StratifiedKFold, train_test_split,
                                       learning_curve, cross_validate)
from sklearn.preprocessing    import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBClassifier;  _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed — XGBoost is co-primary in v18.")

try:
    from imblearn.over_sampling import SMOTE;  _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False
    print("[WARN] imblearn not installed — SMOTE oversampling will be skipped.")

try:
    from lightgbm import LGBMClassifier;     _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier; _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MARKER_SRC_IP   = "2001:db8::ffff"
MARKER_MAGIC    = "IPV6IDS-MARKER-V1"

RANDOM_STATE    = 42
RANDOM_STATE_B  = 123
N_FOLDS         = 5
TEST_SIZE       = 0.20
PRIMARY_K       = 10

XGB_LR_SWEEP    = [0.05, 0.1, 0.2, 0.3]
XGB_DEPTH_SWEEP = [3, 4, 5, 6]
RF_NEST_SWEEP   = [200, 300, 500]
RF_DEPTH_SWEEP  = [5, 7, 10, None]
K_FEAT_SWEEP    = [5, 10, 15]

MIN_CLASS_WINDOWS    = 10
LEAKAGE_AUC_WARN     = 0.98
OVERFIT_GAP_WARN     = 0.05
MAX_SINGLE_FEAT_BACC = 0.999
BOOTSTRAP_N_ITER     = 500
ALWAYS_EXCLUDED_FEATURES = {"ra_rate", "ns_rate"}
BURST_SUBWINDOW = 0.25

DEFAULT_EXTERNAL_COL_MAP = {
    "Flow Packets/s":  "pkt_rate",
    "Fwd IAT Mean":    "mean_iat_ms",
    "Fwd IAT Std":     "std_iat_ms",
    "Fwd IAT Min":     "min_iat_ms",
    "packets_per_sec": "pkt_rate",
    "mean_iat":        "mean_iat_ms",
    "std_iat":         "std_iat_ms",
    "min_iat":         "min_iat_ms",
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE NAME → LABEL  (used by the v4 CSV fallback)
# ─────────────────────────────────────────────────────────────────────────────

def phase_name_to_label(phase_name: str) -> str:
    """Convert a v4 phase name (e.g. 'ra_attack_1') to a class label.
    Used as safety fallback when the events.csv label column contains N/A."""
    p = phase_name.lower()
    if "combined" in p:
        return "Combined_Attack"
    if ("ra" in p or "router" in p) and \
       ("attack" in p or "slow" in p or "flood" in p):
        return "RA_Attack"
    if ("nd" in p or "ns" in p or "solicitat" in p) and \
       ("attack" in p or "slow" in p):
        return "ND_Attack"
    return "Normal"

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117";  PANEL_BG = "#161b22";  CARD_BG = "#1c2333"
ACCENT1  = "#e94560";  ACCENT2  = "#4cc9f0";  ACCENT3 = "#7b2fbe"
ACCENT4  = "#f4a261";  ACCENT5  = "#2ec4b6"
TEXT_COL = "#e6edf3";  GRID_COL = "#21262d";  BORDER  = "#30363d"

CLS_COLORS = {"Normal": "#4cc9f0", "RA_Attack": "#e94560",
              "ND_Attack": "#7b2fbe", "Combined_Attack": "#f4a261"}

LEG_KW = dict(labelcolor=TEXT_COL, facecolor=CARD_BG, edgecolor=BORDER,
              framealpha=1.0, fontsize=9)

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, which="both")
    ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(color=GRID_COL, alpha=0.6, linewidth=0.6)

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig); print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class _SafeXGB(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible XGBClassifier with label re-encoding and
    enhanced regularization (gamma=1.0, reg_lambda=2.0)."""
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.1,
                 eval_metric="mlogloss", reg_alpha=0.1, reg_lambda=2.0,
                 gamma=1.0, colsample_bytree=0.8, subsample=0.8,
                 random_state=42, verbosity=0):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.learning_rate=learning_rate; self.eval_metric=eval_metric
        self.reg_alpha=reg_alpha; self.reg_lambda=reg_lambda
        self.gamma=gamma; self.colsample_bytree=colsample_bytree
        self.subsample=subsample; self.random_state=random_state
        self.verbosity=verbosity

    def fit(self, X, y):
        self._le = LabelEncoder(); y_enc = self._le.fit_transform(y)
        self._xgb = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, eval_metric=self.eval_metric,
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            gamma=self.gamma, colsample_bytree=self.colsample_bytree,
            subsample=self.subsample, random_state=self.random_state,
            verbosity=self.verbosity)
        self._xgb.fit(X, y_enc); self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._xgb.predict(X))

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)

    def _get_global_classes(self):
        return self._le.classes_

    @property
    def feature_importances_(self):
        return self._xgb.feature_importances_


def make_xgb(learning_rate=0.1, max_depth=4):
    return _SafeXGB(n_estimators=200, max_depth=max_depth,
                    learning_rate=learning_rate, eval_metric="mlogloss",
                    reg_alpha=0.1, reg_lambda=2.0, gamma=1.0,
                    colsample_bytree=0.8, subsample=0.8,
                    random_state=RANDOM_STATE, verbosity=0)


def make_rf(n_estimators=200, max_depth=5):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt",
        min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=1)


def safe_predict_proba(clf, X, global_n_cls):
    raw = clf.predict_proba(X)
    learned = (clf._get_global_classes() if hasattr(clf, '_get_global_classes')
               else getattr(clf, 'classes_', None))
    if learned is None or (raw.shape[1] == global_n_cls and len(learned) == global_n_cls):
        return raw
    full = np.zeros((X.shape[0], global_n_cls))
    for li, gc in enumerate(learned):
        if li < raw.shape[1] and gc < global_n_cls:
            full[:, gc] = raw[:, li]
    return full


def max_burst_rate(times, window=1.0):
    if len(times) < 2: return 0.0
    times = sorted(times); max_r = lo = 0
    for hi in range(len(times)):
        while times[hi] - times[lo] > window: lo += 1
        max_r = max(max_r, (hi - lo + 1) / window)
    return max_r


def safe_roc_auc(y_true, y_prob, nc):
    try:
        present = np.unique(y_true)
        if len(present) < 2: return float("nan"), "skipped (only 1 class)"
        if nc == 2: return float(roc_auc_score(y_true, y_prob[:, 1])), "binary"
        if len(present) < nc:
            scores = [roc_auc_score((np.asarray(y_true)==ci).astype(int), y_prob[:,ci])
                      for ci in present
                      if ci < y_prob.shape[1]
                      and (np.asarray(y_true)==ci).sum() > 0
                      and (np.asarray(y_true)!=ci).sum() > 0]
            val = float(np.mean(scores)) if scores else float("nan")
            return val, f"partial OvR ({len(present)}/{nc} classes)"
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                   average="macro")), "macro OvR"
    except Exception as e:
        return float("nan"), f"error: {e}"


def safe_pr_auc(y_true, y_prob, nc):
    try:
        present = np.unique(y_true)
        if len(present) < 2: return float("nan")
        y_bin = label_binarize(y_true, classes=list(range(nc)))
        if y_bin.shape[1] != nc:
            y_bin = np.zeros((len(y_true), nc), dtype=int)
            for i in range(nc): y_bin[:,i] = (np.asarray(y_true)==i).astype(int)
        scores = [average_precision_score(y_bin[:,i], y_prob[:,i])
                  for i in range(nc)
                  if y_bin[:,i].sum() > 0 and (1-y_bin[:,i]).sum() > 0
                  and i < y_prob.shape[1]]
        return float(np.mean(scores)) if scores else float("nan")
    except Exception as e:
        print(f"  [WARN] PR-AUC: {e}"); return float("nan")


def bootstrap_f1_ci(y_true, y_pred, class_names, n_boot=BOOTSTRAP_N_ITER, ci=0.95):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true); y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n_c = len(class_names); macro_buf = []
    class_bufs = {c: [] for c in class_names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n); yt, yp = y_true[idx], y_pred[idx]
        macro_buf.append(f1_score(yt, yp, average="macro", zero_division=0))
        per = f1_score(yt, yp, average=None, labels=list(range(n_c)), zero_division=0)
        for ci_i, cls in enumerate(class_names):
            class_bufs[cls].append(per[ci_i] if ci_i < len(per) else 0.0)
    lo_p, hi_p = (1-ci)/2*100, (1+ci)/2*100
    result = {}
    for cls in class_names:
        a = np.array(class_bufs[cls])
        result[cls] = (float(np.mean(a)), float(np.percentile(a, lo_p)),
                       float(np.percentile(a, hi_p)))
    a = np.array(macro_buf)
    result["macro"] = (float(np.mean(a)), float(np.percentile(a, lo_p)),
                       float(np.percentile(a, hi_p)))
    return result


def print_per_class_f1(y_true, y_pred, le, tag=""):
    classes = list(le.classes_); n_c = len(classes); labels = list(range(n_c))
    precs = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recs  = recall_score(   y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s   = f1_score(       y_true, y_pred, labels=labels, average=None, zero_division=0)
    print(f"  ── Per-class metrics {tag} {'─'*max(0,40-len(tag))}")
    print(f"  {'Class':16s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*8}  {'-'*8}")
    for i, cls in enumerate(classes):
        flag = "  ← LOW" if recs[i] < 0.60 and cls in ("RA_Attack","Combined_Attack") else ""
        print(f"  {cls:16s}  {precs[i]*100:>9.2f}%  {recs[i]*100:>7.2f}%  "
              f"{f1s[i]*100:>7.2f}%{flag}")
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  {'macro':16s}  {'—':>10}  {'—':>8}  {mf1*100:>7.2f}%")
    return f1s, precs, recs


# =============================================================================
# MARKER PARSER — v5 bash script format (in-band ICMPv6 echo packets)
# =============================================================================

def parse_markers_from_pcap(pcap_path):
    """Scan a PCAP for orchestrator marker packets (v5 format).
    Returns (phase_timeline, node_records, marker_count).
    marker_count == 0 signals the caller to try the v4 CSV fallback."""
    phase_starts = {}; phase_timeline = []; marker_count = 0
    node_records = {"router": None, "attacker": None, "victims": []}

    with PcapReader(pcap_path) as pr:
        for pkt in pr:
            if not pkt.haslayer(IPv6) or not pkt.haslayer(ICMPv6EchoRequest):
                continue
            if str(pkt[IPv6].src) != MARKER_SRC_IP: continue
            try:
                payload = bytes(pkt[ICMPv6EchoRequest].data).decode("utf-8","ignore")
            except Exception:
                continue
            if not payload.startswith(MARKER_MAGIC): continue
            marker_count += 1
            parts = payload.split("|", 4)
            if len(parts) < 5: continue
            _, kind, key, label, notes = parts
            ts = float(pkt.time)
            if kind == "phase_start":
                phase_starts[key] = (ts, label, notes)
            elif kind == "phase_end" and key in phase_starts:
                ts_s, lbl, _ = phase_starts.pop(key)
                phase_timeline.append((ts_s, ts, lbl, key))
            elif kind == "node_info":
                fields = notes.split("|")
                if len(fields) >= 3:
                    container, ip, mac = fields[0], fields[1], fields[2].lower()
                    rec = {"container": container, "ip": ip, "mac": mac}
                    if   key == "router":   node_records["router"]   = rec
                    elif key == "attacker": node_records["attacker"] = rec
                    elif key == "victim":   node_records["victims"].append(rec)

    for key, (ts_s, lbl, _) in phase_starts.items():
        phase_timeline.append((ts_s, float("inf"), lbl, key))
    phase_timeline.sort(key=lambda x: x[0])
    return phase_timeline, node_records, marker_count


# =============================================================================
# V4 CSV FALLBACK — reads events.csv + nodes.csv (v4 bash script format)
# =============================================================================

def load_v4_csv_files(events_csv_path, nodes_csv_path):
    """Load the v4 three-file capture format (events.csv + nodes.csv).

    Returns the same tuple as parse_markers_from_pcap() so the rest of
    run_training() works without any further changes:
        (phase_timeline, node_records, n_events)

    n_events == 0 means both files were missing or empty (fatal for training).
    """
    node_records   = {"router": None, "attacker": None, "victims": []}
    phase_timeline = []
    n_events       = 0

    # ── nodes.csv ────────────────────────────────────────────────────────────
    if not Path(nodes_csv_path).exists():
        print(f"  [WARN] nodes.csv not found: {nodes_csv_path}")
    else:
        with open(nodes_csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                role      = row.get("role",      "").strip().lower()
                mac       = row.get("mac",       "").strip().lower()
                ip        = row.get("ipv6", row.get("ip", "")).strip()
                container = row.get("container", "").strip()
                rec = {"container": container, "ip": ip, "mac": mac}
                if   role == "router":   node_records["router"]   = rec
                elif role == "attacker": node_records["attacker"] = rec
                elif role == "victim":   node_records["victims"].append(rec)
        print(f"  Loaded nodes.csv  → "
              f"router={node_records['router'] is not None}, "
              f"attacker={node_records['attacker'] is not None}, "
              f"victims={len(node_records['victims'])}")

    # ── events.csv ───────────────────────────────────────────────────────────
    if not Path(events_csv_path).exists():
        print(f"  [WARN] events.csv not found: {events_csv_path}")
        return phase_timeline, node_records, 0

    VALID_LABELS = {"Normal", "RA_Attack", "ND_Attack", "Combined_Attack"}
    phase_starts = {}

    with open(events_csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            evt   = row.get("event", "").strip()
            phase = row.get("phase", "").strip()
            label = row.get("label", "").strip()
            try:
                ts = float(row.get("ts_epoch", 0))
            except ValueError:
                continue
            n_events += 1

            # Use the label column when it contains a recognised class;
            # otherwise derive from the phase name (handles N/A rows safely)
            lbl = label if label in VALID_LABELS else phase_name_to_label(phase)

            if evt == "phase_start":
                phase_starts[phase] = (ts, lbl)
            elif evt == "phase_end" and phase in phase_starts:
                ts_s, lbl_s = phase_starts.pop(phase)
                phase_timeline.append((ts_s, ts, lbl_s, phase))

    # Unclosed phases extend to infinity
    for phase, (ts_s, lbl) in phase_starts.items():
        phase_timeline.append((ts_s, float("inf"), lbl, phase))

    phase_timeline.sort(key=lambda x: x[0])
    print(f"  Loaded events.csv → {n_events} rows → "
          f"{len(phase_timeline)} phase intervals")
    return phase_timeline, node_records, n_events


def timeline_label(ts, phase_timeline):
    for start, end, label, _ in phase_timeline:
        if start <= ts < end: return label
    return "Normal"


# =============================================================================
# FEATURE EXTRACTION — PCAP → 1-second window DataFrame
# =============================================================================

def extract_windows_from_pcap(pcap_path, phase_timeline, attacker_mac,
                              router_mac="", verbose=True):
    if verbose: print(f"  Parsing PCAP: {pcap_path}")
    pkt_info_list = []; total = ipv6_count = skipped_markers = 0

    with PcapReader(pcap_path) as pr:
        for pkt in pr:
            total += 1
            if not pkt.haslayer(IPv6): continue
            ipv6_count += 1
            src = str(pkt[IPv6].src)
            if src == MARKER_SRC_IP: skipped_markers += 1; continue
            dst = str(pkt[IPv6].dst); ts = float(pkt.time)
            eth_src = pkt[Ether].src.lower() if pkt.haslayer(Ether) else ""
            is_ra = pkt.haslayer(ICMPv6ND_RA); is_ns = pkt.haslayer(ICMPv6ND_NS)
            is_na = pkt.haslayer(ICMPv6ND_NA)
            is_echo = pkt.haslayer(ICMPv6EchoRequest) or pkt.haslayer(ICMPv6EchoReply)
            ns_target = str(getattr(pkt[ICMPv6ND_NS],"tgt","::")) if is_ns else ""
            pkt_info_list.append({"ts":ts,"src":src,"dst":dst,"eth_src":eth_src,
                "pkt_len":len(pkt),"is_ra":int(is_ra),"is_ns":int(is_ns),
                "is_na":int(is_na),"is_echo":int(is_echo),"ns_target":ns_target})

    if verbose:
        print(f"  Total packets       : {total:,}")
        print(f"  IPv6 packets        : {ipv6_count:,}")
        print(f"  Marker packets      : {skipped_markers:,}  (excluded)")
        print(f"  Feature-stream pkts : {len(pkt_info_list):,}")
    if not pkt_info_list:
        print("  [FATAL] No IPv6 packets found."); sys.exit(1)

    buckets = defaultdict(list)
    for info in pkt_info_list: buckets[int(info["ts"])].append(info)
    sorted_bkts   = sorted(buckets.keys())
    bkt_rate_hist = {b: len(buckets[b]) for b in sorted_bkts}
    have_timeline = len(phase_timeline) > 0
    window_rows   = []

    for b in sorted_bkts:
        pkts = buckets[b]; n = len(pkts)
        ts_arr = sorted(p["ts"] for p in pkts)
        if n >= 2:
            iats   = [max((ts_arr[i+1]-ts_arr[i])*1000.0, 1e-6) for i in range(n-1)]
            m_iat  = float(np.mean(iats)); s_iat = float(np.std(iats))
            mi_iat = float(np.min(iats))
        else:
            m_iat = s_iat = mi_iat = 0.0
        burstiness  = s_iat / (m_iat + 1e-6)
        ra_cnt = sum(p["is_ra"] for p in pkts); ns_cnt = sum(p["is_ns"] for p in pkts)
        na_cnt = sum(p["is_na"] for p in pkts)
        ra_burst_rate = max_burst_rate([p["ts"] for p in pkts if p["is_ra"]], BURST_SUBWINDOW)
        ns_burst_rate = max_burst_rate([p["ts"] for p in pkts if p["is_ns"]], BURST_SUBWINDOW)
        src_set = {p["src"] for p in pkts}; dst_set = {p["dst"] for p in pkts}
        mc_cnt  = sum(1 for p in pkts if p["dst"].startswith("ff"))
        ra_src_set = {p["src"] for p in pkts if p["is_ra"]}
        ns_tgt_set = {p["ns_target"] for p in pkts if p["is_ns"] and p["ns_target"]}
        multicast_ratio = mc_cnt / n
        src_diversity   = len(src_set) / n
        ra_src_div      = len(ra_src_set) / max(ra_cnt, 1)
        ns_unique_tgt_r = len(ns_tgt_set) / max(ns_cnt, 1)
        s3t  = sum(bkt_rate_hist.get(b-i, 0) for i in range(1, 4))
        s3nd = sum(sum(p["is_ra"]+p["is_ns"]+p["is_na"] for p in buckets.get(b-i,[]))
                   for i in range(1, 4))
        sliding_nd_ratio_3s = s3nd / max(s3t, 1)

        if have_timeline:
            label = timeline_label(float(b), phase_timeline)
        elif attacker_mac:
            has_ra = any(p["eth_src"]==attacker_mac and p["is_ra"] for p in pkts)
            has_ns = any(p["eth_src"]==attacker_mac and p["is_ns"] for p in pkts)
            label  = "RA_Attack" if has_ra else "ND_Attack" if has_ns else "Normal"
        else:
            label = "Normal"

        window_rows.append({
            "time_bucket":b, "pkt_rate":float(n),
            "ra_rate":float(ra_cnt), "ns_rate":float(ns_cnt), "na_rate":float(na_cnt),
            "ra_burst_rate":ra_burst_rate, "ns_burst_rate":ns_burst_rate,
            "unique_src_count":float(len(src_set)), "unique_dst_count":float(len(dst_set)),
            "mean_iat_ms":m_iat, "std_iat_ms":s_iat, "min_iat_ms":mi_iat,
            "burstiness":burstiness, "multicast_ratio":multicast_ratio,
            "src_diversity":src_diversity, "ra_src_diversity":ra_src_div,
            "ns_unique_tgt_r":ns_unique_tgt_r, "sliding_nd_ratio_3s":sliding_nd_ratio_3s,
            "label":label,
        })

    df_win = pd.DataFrame(window_rows)
    if verbose: print(f"  Window DataFrame    : {df_win.shape}")
    return df_win


# =============================================================================
# EXTERNAL CSV LOADER
# =============================================================================

def load_external_csv(csv_path, internal_feature_cols, label_col="label",
                      label_map=None, col_map=None):
    print(f"  Loading external CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw CSV shape       : {df.shape}")

    col_map = {**DEFAULT_EXTERNAL_COL_MAP, **(col_map or {})}
    rename_applied = {k: v for k, v in col_map.items() if k in df.columns}
    if rename_applied:
        df = df.rename(columns=rename_applied)
        print(f"  Renamed columns     : {rename_applied}")

    if label_col not in df.columns:
        for alt in ("Label","class","Class","category","Category"):
            if alt in df.columns:
                print(f"  Note: using '{alt}' as label column."); label_col = alt; break
        else:
            raise ValueError(f"Label column '{label_col}' not found. "
                             f"Available: {list(df.columns)}")

    missing = [f for f in internal_feature_cols if f not in df.columns]
    for f in missing: df[f] = 0.0
    if missing:
        print(f"  [WARN] External CSV missing {len(missing)}/{len(internal_feature_cols)} features.")
        print(f"         Filled with zeros: {missing}")
        print(f"         Model evaluated on DEGRADED feature set — report this in thesis.")

    df["label"] = (df[label_col].astype(str).map(label_map).fillna("Normal")
                   if label_map else df[label_col].astype(str))

    print(f"  External label distribution:")
    for cls, n in df["label"].value_counts().items():
        print(f"    {cls:24s} : {n:,}")
    return df[internal_feature_cols + ["label"]].copy()


# =============================================================================
# MODE: TRAIN
# =============================================================================

def run_training(args):
    print("=" * 70)
    print("MODE: TRAIN")
    print("=" * 70)
    print(f"  PCAP            : {args.pcap}")
    print(f"  Model bundle out: {args.model_out}")

    if not Path(args.pcap).exists():
        print(f"\n  [FATAL] PCAP not found: {args.pcap}"); sys.exit(1)

    # ── STEP 1: Parse markers (v5) — fall back to v4 CSVs if none found ──────
    print("\n" + "─" * 70)
    print("STEP 1 — Parsing in-band markers from PCAP")
    print("─" * 70)
    phase_timeline, node_records, marker_count = parse_markers_from_pcap(args.pcap)
    print(f"  Marker packets found: {marker_count}")
    print(f"  Phase intervals     : {len(phase_timeline)}")

    if marker_count == 0:
        # ── v4 CSV FALLBACK ──────────────────────────────────────────────────
        print("\n  [WARN] No in-band markers found — PCAP was captured with the")
        print("         v4 bash script. Trying v4 CSV fallback ...")

        pcap_dir  = Path(args.pcap).parent
        pcap_stem = Path(args.pcap).stem      # e.g. "raw_capture"

        def _find_csv(flag_val, suffix):
            """Return the best available path for a companion CSV file."""
            if flag_val and Path(flag_val).exists():
                return flag_val
            beside = pcap_dir / f"{pcap_stem}{suffix}"
            if beside.exists(): return str(beside)
            cwd = Path(f"./{pcap_stem}{suffix}")
            if cwd.exists(): return str(cwd)
            return str(beside)   # return expected path so error message is clear

        events_csv = _find_csv(getattr(args, "events_csv", None), "_events.csv")
        nodes_csv  = _find_csv(getattr(args, "nodes_csv",  None), "_nodes.csv")

        print(f"\n  events.csv path : {events_csv}")
        print(f"  nodes.csv  path : {nodes_csv}")

        phase_timeline, node_records, marker_count = \
            load_v4_csv_files(events_csv, nodes_csv)

        if marker_count == 0:
            print("\n  [FATAL] Could not load phase data from events.csv.")
            print("          Ensure these files exist beside the PCAP:")
            print(f"            {pcap_dir / (pcap_stem + '_events.csv')}")
            print(f"            {pcap_dir / (pcap_stem + '_nodes.csv')}")
            print("          Or pass them explicitly:")
            print("            --events-csv /full/path/to/raw_capture_events.csv")
            print("            --nodes-csv  /full/path/to/raw_capture_nodes.csv")
            sys.exit(1)

        print(f"\n  v4 CSV fallback succeeded — "
              f"{len(phase_timeline)} phase intervals loaded.")

    if node_records["attacker"] is None:
        print("\n  [FATAL] Attacker entry missing from nodes data."); sys.exit(1)

    ATTACKER_MAC = node_records["attacker"]["mac"]
    ROUTER_MAC   = node_records["router"]["mac"] if node_records["router"] else ""
    print(f"\n  Attacker MAC : {ATTACKER_MAC}")
    print(f"  Router   MAC : {ROUTER_MAC}")
    print(f"  Victims      : {len(node_records['victims'])}")

    print("\n  Phase timeline:")
    for s, e, lbl, pname in phase_timeline:
        e_str = f"{e:.1f}" if e != float("inf") else "∞"
        print(f"    {s:.1f} → {e_str}  [{lbl:16s}]  ({pname})")

    # ── STEP 2: Feature extraction ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2 — Window feature extraction")
    print("─" * 70)
    df_win = extract_windows_from_pcap(args.pcap, phase_timeline,
                                       ATTACKER_MAC, ROUTER_MAC)
    vc_win = df_win["label"].value_counts()
    print("\n  Window class distribution:")
    for cls in ["Normal","RA_Attack","ND_Attack","Combined_Attack"]:
        n = vc_win.get(cls, 0); pct = 100*n/len(df_win) if len(df_win) else 0
        print(f"    {cls:16s}: {n:5,d} windows  ({pct:.1f}%)")

    # ── STEP 3: Class balance assertion ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3 — Class balance assertion")
    print("─" * 70)
    class_counts = {cls: int(vc_win.get(cls,0))
        for cls in ["Normal","RA_Attack","ND_Attack","Combined_Attack"]}
    missing_cls = [c for c,n in class_counts.items() if n < MIN_CLASS_WINDOWS]
    for cls, n in class_counts.items():
        print(f"    {cls:16s}: "
              f"{'OK (' + str(n) + ')' if n >= MIN_CLASS_WINDOWS else 'FAIL — only ' + str(n)}")
    if missing_cls:
        print(f"\n  [FATAL] Insufficient windows: {missing_cls}"); sys.exit(1)

    # ── STEP 4: Preprocessing + Stratified splits ────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4 — ML preprocessing (Stratified Splits)")
    print("─" * 70)
    _exclude = {"time_bucket","label"} | ALWAYS_EXCLUDED_FEATURES
    FEATURE_COLS = [c for c in df_win.columns if c not in _exclude
                    and df_win[c].nunique() > 1]
    print(f"  Feature pool ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    X_raw  = df_win[FEATURE_COLS].values.astype(float)
    le     = LabelEncoder()
    y      = le.fit_transform(df_win["label"])
    n_cls  = len(le.classes_)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"  Classes      : {list(le.classes_)}")
    print(f"  Total windows: {len(y):,}")

    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"\n  Split A (rs={RANDOM_STATE}): train={len(y_train_raw):,}  test={len(y_test):,}")

    majority_n = max(Counter(y_train_raw).values())
    needs_smote = any(v < int(majority_n*0.15) for v in Counter(y_train_raw).values())
    if _HAS_SMOTE and needs_smote:
        k = max(1, min(5, min(Counter(y_train_raw).values()) - 1))
        X_train, y_train = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(
            X_train_raw, y_train_raw)
        print(f"  [SMOTE] {len(y_train_raw):,} → {len(y_train):,}  (k={k})")
    else:
        X_train, y_train = X_train_raw, y_train_raw
        print(f"  [SMOTE] {'Not available' if not _HAS_SMOTE else 'Skipped (balanced)'}.")

    X_tr_b_raw, X_te_b, y_tr_b_raw, y_te_b = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE_B)
    if _HAS_SMOTE and needs_smote:
        min_b = min(Counter(y_tr_b_raw).values())
        if min_b >= 2:
            X_tr_b, y_tr_b = SMOTE(random_state=RANDOM_STATE_B,
                                    k_neighbors=max(1, min(5, min_b-1))).fit_resample(
                X_tr_b_raw, y_tr_b_raw)
        else:
            X_tr_b, y_tr_b = X_tr_b_raw, y_tr_b_raw
    else:
        X_tr_b, y_tr_b = X_tr_b_raw, y_tr_b_raw

    # ── STEP 5: ANOVA-F ranking ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5 — ANOVA-F feature ranking (train-only)")
    print("─" * 70)
    skb = SelectKBest(score_func=f_classif, k="all")
    skb.fit(X_train, y_train)
    ranking         = np.argsort(skb.scores_)[::-1]
    ranked_features = [FEATURE_COLS[i] for i in ranking]
    ranked_scores   = [skb.scores_[i]  for i in ranking]
    ranked_pvals    = [skb.pvalues_[i] for i in ranking]
    print(f"\n  {'Rank':>5}  {'Feature':28s}  {'F-score':>10}  {'p-value':>12}")
    for rank, (feat, fs, pv) in enumerate(
            zip(ranked_features, ranked_scores, ranked_pvals), 1):
        print(f"  {rank:>5}{'★' if rank<=PRIMARY_K else ' '} "
              f"{feat:28s}  {fs:>10.3f}  {pv:>12.2e}")
    TOP_K_FEATURES = ranked_features[:min(PRIMARY_K, len(FEATURE_COLS))]
    TOP_K_IDX      = [FEATURE_COLS.index(f) for f in TOP_K_FEATURES]
    print(f"\n  Top-{len(TOP_K_FEATURES)}: {TOP_K_FEATURES}")

    # ── STEP 6: Single-feature leakage scan ──────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 6 — Single-feature leakage scan")
    print("─" * 70)
    high_solo = []
    print(f"  {'Feature':30s}  {'ROC-AUC':>8}  {'BalAcc':>8}")
    for fi, feat in enumerate(FEATURE_COLS):
        clf1 = make_rf(50, 3)
        clf1.fit(X_train[:, fi:fi+1], y_train)
        prob1 = safe_predict_proba(clf1, X_test[:, fi:fi+1], n_cls)
        pred1 = clf1.predict(X_test[:, fi:fi+1])
        auc1, _ = safe_roc_auc(y_test, prob1, n_cls)
        bacc1   = balanced_accuracy_score(y_test, pred1)
        flags   = []
        if not math.isnan(auc1) and auc1 > LEAKAGE_AUC_WARN: flags.append("AUC-WARN")
        if bacc1 > MAX_SINGLE_FEAT_BACC: high_solo.append(feat); flags.append("BACC-EXCLUDED")
        auc_s = f"{auc1:>8.4f}" if not math.isnan(auc1) else "     NaN"
        flag_s = (" ← " + " | ".join(flags)) if flags else ""
        print(f"  {feat:30s}  {auc_s}  {bacc1*100:>7.2f}%{flag_s}")
    if high_solo:
        print(f"\n  [BACC-CAP] Excluding: {high_solo}")
        keep = [c not in high_solo for c in FEATURE_COLS]
        ki   = [i for i,k in enumerate(keep) if k]
        FEATURE_COLS    = [c for c,k in zip(FEATURE_COLS, keep) if k]
        X_train         = X_train[:, ki];  X_test  = X_test[:, ki]
        X_tr_b          = X_tr_b[:, ki];   X_te_b  = X_te_b[:, ki]
        X_scaled        = X_scaled[:, ki]
        TOP_K_FEATURES  = [f for f in TOP_K_FEATURES if f in FEATURE_COLS]
        TOP_K_IDX       = [FEATURE_COLS.index(f) for f in TOP_K_FEATURES]
        ranked_features = [f for f in ranked_features if f in FEATURE_COLS]

    # ── STEP 7: XGBoost sweep ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 7 — XGBoost hyperparameter sweep")
    print("─" * 70)
    X_tr_k = X_train[:, TOP_K_IDX]; X_te_k = X_test[:, TOP_K_IDX]
    best_xgb_lr = 0.1; best_xgb_depth = 4; clf_xgb_final = None

    if _HAS_XGB:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        xgb_res = []
        print(f"  {'LR':>5}  {'Depth':>5}  {'CV-BalAcc':>10}  {'Test-BalAcc':>11}  {'MacroF1':>8}")
        for lr in XGB_LR_SWEEP:
            for depth in XGB_DEPTH_SWEEP:
                cv_b = [balanced_accuracy_score(y_train[vi], make_xgb(lr,depth).fit(
                    X_tr_k[ti], y_train[ti]).predict(X_tr_k[vi]))
                    for ti, vi in skf.split(X_tr_k, y_train)]
                cv_ba = float(np.mean(cv_b))
                clf_f = make_xgb(lr, depth); clf_f.fit(X_tr_k, y_train)
                pt = clf_f.predict(X_te_k)
                ba  = balanced_accuracy_score(y_test, pt)
                mf1 = f1_score(y_test, pt, average="macro", zero_division=0)
                xgb_res.append({"lr":lr,"depth":depth,"cv_bacc":cv_ba,
                                 "test_bacc":ba,"mf1":mf1,"model":clf_f})
                print(f"  {lr:>5.2f}  {depth:>5}  {cv_ba*100:>9.2f}%  "
                      f"{ba*100:>10.2f}%  {mf1*100:>7.2f}%")
        best = xgb_res[max(range(len(xgb_res)), key=lambda i: xgb_res[i]["cv_bacc"])]
        best_xgb_lr=best["lr"]; best_xgb_depth=best["depth"]; clf_xgb_final=best["model"]
        print(f"\n  ★ Best XGBoost: LR={best_xgb_lr}, depth={best_xgb_depth}")
    else:
        print("  [SKIP] xgboost not installed.")

    # ── STEP 8: Random Forest sweep ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 8 — Random Forest hyperparameter sweep")
    print("─" * 70)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rf_res = []
    print(f"  {'N_est':>5}  {'Depth':>5}  {'CV-BalAcc':>10}  {'Test-BalAcc':>11}  {'MacroF1':>8}")
    for n_est in RF_NEST_SWEEP:
        for depth in RF_DEPTH_SWEEP:
            cv_b = [balanced_accuracy_score(y_train[vi], make_rf(n_est,depth).fit(
                X_tr_k[ti], y_train[ti]).predict(X_tr_k[vi]))
                for ti, vi in skf.split(X_tr_k, y_train)]
            cv_ba = float(np.mean(cv_b))
            clf_f = make_rf(n_est, depth); clf_f.fit(X_tr_k, y_train)
            pt = clf_f.predict(X_te_k)
            ba  = balanced_accuracy_score(y_test, pt)
            mf1 = f1_score(y_test, pt, average="macro", zero_division=0)
            d_s = str(depth) if depth is not None else "None"
            rf_res.append({"n_est":n_est,"depth":depth,"cv_bacc":cv_ba,
                            "test_bacc":ba,"mf1":mf1,"model":clf_f})
            print(f"  {n_est:>5}  {d_s:>5}  {cv_ba*100:>9.2f}%  "
                  f"{ba*100:>10.2f}%  {mf1*100:>7.2f}%")
    best = rf_res[max(range(len(rf_res)), key=lambda i: rf_res[i]["cv_bacc"])]
    best_rf_nest=best["n_est"]; best_rf_depth=best["depth"]; clf_rf_final=best["model"]
    d_s = str(best_rf_depth) if best_rf_depth is not None else "None"
    print(f"\n  ★ Best RF: n_estimators={best_rf_nest}, depth={d_s}")

    # ── STEP 9: Final evaluation (Split A) ───────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 9 — Final evaluation (Split A)")
    print("─" * 70)
    final_models = {}
    for mname, clf_f in [("XGBoost", clf_xgb_final), ("RF", clf_rf_final)]:
        if clf_f is None: continue
        pred_te = clf_f.predict(X_te_k); pred_tr = clf_f.predict(X_tr_k)
        prob_te = safe_predict_proba(clf_f, X_te_k, n_cls)
        acc_te  = accuracy_score(y_test, pred_te)
        acc_tr  = accuracy_score(y_train, pred_tr)
        bacc_te = balanced_accuracy_score(y_test, pred_te)
        mf1_te  = f1_score(y_test, pred_te, average="macro", zero_division=0)
        gap     = acc_tr - acc_te
        roc, roc_note = safe_roc_auc(y_test, prob_te, n_cls)
        pr_auc  = safe_pr_auc(y_test, prob_te, n_cls)
        final_models[mname] = {"clf":clf_f,"pred":pred_te,"prob":prob_te,
            "acc_tr":acc_tr,"acc_te":acc_te,"bacc":bacc_te,"mf1":mf1_te,
            "gap":gap,"roc":roc,"pr_auc":pr_auc}
        print(f"\n  ── {mname} ──")
        print(f"  Train Accuracy    : {acc_tr*100:.2f}%")
        print(f"  Test  Accuracy    : {acc_te*100:.2f}%")
        print(f"  Overfitting gap   : {gap*100:+.2f}%")
        print(f"  Balanced Accuracy : {bacc_te*100:.2f}%")
        print(f"  Macro F1          : {mf1_te*100:.2f}%")
        print(f"  ROC-AUC           : {roc:.4f}  [{roc_note}]")
        print(f"  PR-AUC            : {pr_auc:.4f}")
        print_per_class_f1(y_test, pred_te, le, tag=f"[{mname} A]")

    # ── STEP 10: Split B + 5-fold CV ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 10 — Split B + 5-fold CV stability")
    print("─" * 70)
    X_tr_bk = X_tr_b[:, TOP_K_IDX]; X_te_bk = X_te_b[:, TOP_K_IDX]
    for mname, factory in [
        ("XGBoost", lambda: make_xgb(best_xgb_lr, best_xgb_depth) if _HAS_XGB else None),
        ("RF",      lambda: make_rf(best_rf_nest, best_rf_depth)),
    ]:
        clf_t = factory()
        if clf_t is None: continue
        clf_t.fit(X_tr_bk, y_tr_b)
        pred_t = clf_t.predict(X_te_bk)
        bacc_t = balanced_accuracy_score(y_te_b, pred_t)
        mf1_t  = f1_score(y_te_b, pred_t, average="macro", zero_division=0)
        cv_res = cross_validate(factory(), X_scaled[:, TOP_K_IDX], y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring={"bacc":"balanced_accuracy","f1":"f1_macro"},
            return_train_score=True, n_jobs=1)
        cv_mean = float(np.mean(cv_res["test_bacc"]))
        cv_std  = float(np.std (cv_res["test_bacc"]))
        delta   = final_models.get(mname, {}).get("bacc", bacc_t) - bacc_t
        print(f"\n  ── {mname} (Split B) ──")
        print(f"  BalAcc            : {bacc_t*100:.2f}%")
        print(f"  Macro F1          : {mf1_t*100:.2f}%")
        print(f"  Variance (A vs B) : {delta*100:+.2f}%")
        print(f"  5-Fold CV         : {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

    # ── STEP 11: Bootstrap 95% CI ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"STEP 11 — Bootstrap 95% CI ({BOOTSTRAP_N_ITER} iters)")
    print("─" * 70)
    primary = "XGBoost" if "XGBoost" in final_models else "RF"
    ci_res  = bootstrap_f1_ci(y_test, final_models[primary]["pred"], list(le.classes_))
    print(f"\n  Using {primary} on Split A.")
    print(f"  {'Class':16s}  {'Mean F1':>8}  {'95% lo':>8}  {'95% hi':>8}")
    for cls in list(le.classes_) + ["macro"]:
        m, lo, hi = ci_res[cls]
        print(f"  {cls:16s}  {m*100:>7.2f}%  {lo*100:>7.2f}%  {hi*100:>7.2f}%")

    # ── STEP 12: Dummy classifier sanity ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 12 — Dummy classifier sanity check")
    print("─" * 70)
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_tr_k, y_train)
    bacc_d = balanced_accuracy_score(y_test, dummy.predict(X_te_k))
    for mname, res in final_models.items():
        delta = res["bacc"] - bacc_d
        print(f"  {mname:10s}  BalAcc={res['bacc']*100:.2f}%  "
              f"(Dummy={bacc_d*100:.2f}%, Δ={delta*100:+.1f}pp)")

    # ── STEP 13: Save model bundle ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 13 — Saving model bundle")
    print("─" * 70)
    bundle = {
        "version": "v18",
        "scaler": scaler, "label_encoder": le,
        "feature_cols": FEATURE_COLS, "top_k_features": TOP_K_FEATURES,
        "top_k_idx": TOP_K_IDX, "xgb": clf_xgb_final, "rf": clf_rf_final,
        "best_xgb_lr": best_xgb_lr, "best_xgb_depth": best_xgb_depth,
        "best_rf_nest": best_rf_nest, "best_rf_depth": best_rf_depth,
        "n_cls": n_cls,
        "training_metrics": {
            mn: {"bacc": float(r["bacc"]), "mf1": float(r["mf1"]),
                 "roc": float(r["roc"]), "acc_te": float(r["acc_te"])}
            for mn, r in final_models.items()
        }
    }
    joblib.dump(bundle, args.model_out)
    print(f"  Bundle saved → {args.model_out}")

    # ── STEP 14: Plots ────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n" + "─" * 70)
        print("STEP 14 — Generating visualisations")
        print("─" * 70)
        _generate_training_plots(
            df_win, le, FEATURE_COLS, TOP_K_FEATURES, TOP_K_IDX,
            ranked_features, ranked_scores, X_train_raw, X_test, y_test,
            X_tr_k, X_te_k, y_train, final_models,
            best_xgb_lr, best_xgb_depth, best_rf_nest, best_rf_depth, n_cls)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Bundle : {args.model_out}")
    for mn, r in final_models.items():
        print(f"  {mn:10s} BalAcc: {r['bacc']*100:.2f}%")
    print(f"\n  Next step — external validation:")
    print(f"    python ids_pipeline_v18.py --mode evaluate-csv \\")
    print(f"        --csv Labdataset.csv --label-column <col_name>")


# =============================================================================
# MODE: EVALUATE EXTERNAL DATA
# =============================================================================

def evaluate_external(df_win, bundle, source_label=""):
    print("\n" + "─" * 70)
    print(f"External evaluation — source: {source_label}")
    print("─" * 70)
    scaler  = bundle["scaler"]; le = bundle["label_encoder"]
    feat    = bundle["feature_cols"]; top_idx = bundle["top_k_idx"]
    n_cls   = bundle["n_cls"]

    X_top = scaler.transform(df_win[feat].values.astype(float))[:, top_idx]
    lmap  = {c: i for i, c in enumerate(le.classes_)}
    y_true = np.array([lmap.get(l, -1) for l in df_win["label"]])
    keep   = y_true >= 0
    if keep.sum() == 0:
        print("  [WARN] No external samples map to known internal classes.")
        print(f"  Internal : {list(le.classes_)}")
        print(f"  External : {df_win['label'].unique().tolist()}")
        print("  Use --label-map to remap."); return
    if keep.sum() < len(keep):
        print(f"  [WARN] Dropped {(~keep).sum():,} samples with unknown labels.")
    X_top, y_true = X_top[keep], y_true[keep]

    print(f"\n  Samples evaluated  : {len(y_true):,}")
    print(f"  Present in external: "
          f"{sorted([le.classes_[i] for i in np.unique(y_true)])}")

    results = []
    for mname in ("xgb", "rf"):
        clf = bundle.get(mname)
        if clf is None: continue
        pred = clf.predict(X_top)
        prob = safe_predict_proba(clf, X_top, n_cls)
        bacc = balanced_accuracy_score(y_true, pred)
        mf1  = f1_score(y_true, pred, average="macro", zero_division=0)
        acc  = accuracy_score(y_true, pred)
        roc, roc_note = safe_roc_auc(y_true, prob, n_cls)
        pr_a = safe_pr_auc(y_true, prob, n_cls)
        print(f"\n  ── {mname.upper()} ──")
        print(f"  Accuracy          : {acc*100:.2f}%")
        print(f"  Balanced Accuracy : {bacc*100:.2f}%")
        print(f"  Macro F1          : {mf1*100:.2f}%")
        print(f"  ROC-AUC           : {roc:.4f}  [{roc_note}]")
        print(f"  PR-AUC            : {pr_a:.4f}")
        print_per_class_f1(y_true, pred, le, tag=f"[{mname.upper()} EXT]")
        results.append((source_label, mname, acc, bacc, mf1, roc, pr_a))
        try:
            present = sorted(set(y_true)|set(pred))
            names   = [le.classes_[i] for i in present]
            cm_val  = confusion_matrix(y_true, pred, labels=present)
            fig, ax = plt.subplots(figsize=(7,6))
            fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)
            ConfusionMatrixDisplay(cm_val, display_labels=names).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{mname.upper()} — External: {source_label}", color=TEXT_COL)
            ax.tick_params(colors=TEXT_COL)
            for txt in ax.texts: txt.set_color(TEXT_COL)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                     color=TEXT_COL, fontsize=8)
            plt.setp(ax.get_yticklabels(), color=TEXT_COL)
            safe_l = "".join(c if c.isalnum() else "_" for c in source_label)[:40]
            savefig(fig, f"external_cm_{mname}_{safe_l}.png")
        except Exception as e:
            print(f"  [WARN] CM plot failed: {e}")

    rp = "./external_validation_results.csv"
    write_hdr = not Path(rp).exists()
    with open(rp, "a") as fh:
        if write_hdr: fh.write("source,model,accuracy,bacc,macro_f1,roc_auc,pr_auc\n")
        for row in results: fh.write(",".join(str(x) for x in row) + "\n")
    print(f"\n  Results appended to {rp}")


def run_evaluate_pcap(args):
    print("=" * 70); print("MODE: EVALUATE-PCAP"); print("=" * 70)
    print(f"  External PCAP    : {args.pcap}")
    print(f"  Model bundle in  : {args.model_in}")
    if not Path(args.pcap).exists():
        print(f"\n  [FATAL] PCAP not found: {args.pcap}"); sys.exit(1)
    if not Path(args.model_in).exists():
        print(f"\n  [FATAL] Bundle not found: {args.model_in}"); sys.exit(1)
    bundle = joblib.load(args.model_in)
    print(f"  Bundle version   : {bundle.get('version','?')}")
    print(f"  Internal classes : {list(bundle['label_encoder'].classes_)}")
    print("\n  Scanning for markers ...")
    phase_timeline, _, marker_count = parse_markers_from_pcap(args.pcap)
    if marker_count > 0:
        print(f"  Found {marker_count} markers — using marker-derived labels.")
    else:
        print("  No markers found — using --external-attacker-mac.")
        if not args.external_attacker_mac:
            print("  [WARN] No markers AND no MAC — all windows labelled Normal.")
    attacker_mac = (args.external_attacker_mac or "").lower()
    df_win = extract_windows_from_pcap(args.pcap, phase_timeline, attacker_mac)
    for cls, n in df_win["label"].value_counts().items():
        print(f"    {cls:24s}: {n:,}")
    evaluate_external(df_win, bundle, source_label=Path(args.pcap).name)


def run_evaluate_csv(args):
    print("=" * 70); print("MODE: EVALUATE-CSV"); print("=" * 70)
    print(f"  External CSV     : {args.csv}")
    print(f"  Model bundle in  : {args.model_in}")
    print(f"  Label column     : {args.label_column}")
    if not Path(args.csv).exists():
        print(f"\n  [FATAL] CSV not found: {args.csv}"); sys.exit(1)
    if not Path(args.model_in).exists():
        print(f"\n  [FATAL] Bundle not found: {args.model_in}"); sys.exit(1)
    bundle    = joblib.load(args.model_in)
    label_map = json.loads(args.label_map)  if args.label_map  else None
    col_map   = json.loads(args.column_map) if args.column_map else None
    df_win = load_external_csv(args.csv, bundle["feature_cols"],
                               args.label_column, label_map, col_map)
    evaluate_external(df_win, bundle, source_label=Path(args.csv).name)


# =============================================================================
# PLOTS (train mode)
# =============================================================================

def _generate_training_plots(df_win, le, FEATURE_COLS, TOP_K_FEATURES, TOP_K_IDX,
                              ranked_features, ranked_scores, X_train_raw,
                              X_test, y_test, X_tr_k, X_te_k, y_train,
                              final_models, best_xgb_lr, best_xgb_depth,
                              best_rf_nest, best_rf_depth, n_cls):
    cls_names   = list(le.classes_)
    cls_palette = [CLS_COLORS.get(c, ACCENT2) for c in cls_names]

    print("  PLOT 1 — selectkbest_ranking.png")
    fig, ax = plt.subplots(figsize=(10,7)); fig.patch.set_facecolor(DARK_BG); style_ax(ax)
    bar_cols = [ACCENT1 if f in TOP_K_FEATURES else ACCENT2 for f in ranked_features]
    ax.barh(ranked_features[::-1], ranked_scores[::-1], color=bar_cols[::-1], alpha=0.88)
    ax.set_xlabel("ANOVA-F Score")
    ax.set_title(f"Feature Ranking (★ red = top-{len(TOP_K_FEATURES)})", color=TEXT_COL)
    ax.legend(handles=[mpatches.Patch(color=ACCENT1, label=f"Top-{len(TOP_K_FEATURES)}"),
                        mpatches.Patch(color=ACCENT2, label="Not used")], **LEG_KW)
    plt.tight_layout(); savefig(fig, "selectkbest_ranking.png")

    print("  PLOT 2 — feature_correlation.png")
    top8 = TOP_K_FEATURES[:min(8,len(TOP_K_FEATURES))]
    idx8 = [FEATURE_COLS.index(f) for f in top8]
    corr = pd.DataFrame(X_train_raw[:,idx8], columns=top8).corr().values
    fig, ax = plt.subplots(figsize=(9,8)); fig.patch.set_facecolor(DARK_BG); style_ax(ax)
    cmap = LinearSegmentedColormap.from_list("ids",[ACCENT3,PANEL_BG,ACCENT1])
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(top8))); ax.set_yticks(range(len(top8)))
    ax.set_xticklabels(top8, rotation=45, ha="right", color=TEXT_COL, fontsize=8)
    ax.set_yticklabels(top8, color=TEXT_COL, fontsize=8)
    ax.set_title("Feature Correlation (Top-8)", color=TEXT_COL)
    for i in range(len(top8)):
        for j in range(len(top8)):
            ax.text(j,i,f"{corr[i,j]:.2f}",ha="center",va="center",color=TEXT_COL,fontsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL); cb.outline.set_edgecolor(BORDER)
    plt.tight_layout(); savefig(fig, "feature_correlation.png")

    print("  PLOT 3 — confusion_matrices.png")
    cm_list = [(m, final_models[m]["pred"]) for m in final_models]
    n_cm = len(cm_list)
    fig, axes = plt.subplots(1, n_cm, figsize=(7*n_cm,6)); fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Confusion Matrices — Internal Test Set (Split A)", color=TEXT_COL, fontsize=13)
    if n_cm == 1: axes = [axes]
    for ax, (mn, pred) in zip(axes, cm_list):
        ax.set_facecolor(PANEL_BG)
        present = sorted(set(y_test)|set(pred))
        names   = [le.classes_[i] for i in present]
        cm_val  = confusion_matrix(y_test, pred, labels=present)
        ConfusionMatrixDisplay(cm_val, display_labels=names).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(mn, color=TEXT_COL, fontsize=11)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", color=TEXT_COL, fontsize=8)
        plt.setp(ax.get_yticklabels(), color=TEXT_COL)
        cm_norm = cm_val.astype(float) / (cm_val.max() + 1e-9)
        for txt in ax.texts:
            ci = int(round(txt.get_position()[0])); ri = int(round(txt.get_position()[1]))
            if 0<=ri<cm_norm.shape[0] and 0<=ci<cm_norm.shape[1]:
                txt.set_color("#0d1117" if cm_norm[ri,ci] > 0.45 else "#e6edf3")
            else:
                txt.set_color("#e6edf3")
            txt.set_fontsize(11); txt.set_fontweight("bold")
    plt.tight_layout(); savefig(fig, "confusion_matrices.png")

    print("  PLOT 4 — learning_curve.png")
    lc_data = {}
    lc_cv = StratifiedKFold(n_splits=5, shuffle=False)
    for mname, factory in [
        ("XGBoost", lambda: make_xgb(best_xgb_lr,best_xgb_depth) if _HAS_XGB else None),
        ("RF",      lambda: make_rf(best_rf_nest,best_rf_depth)),
    ]:
        est = factory()
        if est is None: continue
        try:
            ts, tr_sc, va_sc = learning_curve(est, X_tr_k, y_train, cv=lc_cv,
                train_sizes=np.linspace(0.10,1.0,9), scoring="balanced_accuracy", n_jobs=1)
            lc_data[mname] = {"sizes":ts.tolist(),
                               "train":np.mean(tr_sc,axis=1).tolist(),
                               "val":  np.mean(va_sc,axis=1).tolist()}
        except Exception as e:
            print(f"    [WARN] LC {mname}: {e}")
    if lc_data:
        fig, axes = plt.subplots(1,len(lc_data),figsize=(7*len(lc_data),6))
        fig.patch.set_facecolor(DARK_BG)
        if len(lc_data)==1: axes=[axes]
        for ax,(mn,lcd) in zip(axes,lc_data.items()):
            style_ax(ax)
            ax.plot(lcd["sizes"],[v*100 for v in lcd["train"]],"-o",color=ACCENT1,lw=2,label="Train BalAcc")
            ax.plot(lcd["sizes"],[v*100 for v in lcd["val"]],  "-s",color=ACCENT2,lw=2,label="Val BalAcc")
            ax.set_xlabel("Training Set Size"); ax.set_ylabel("Balanced Accuracy (%)")
            ax.set_title(mn,color=TEXT_COL); ax.set_ylim(30,105); ax.legend(**LEG_KW)
        plt.tight_layout(); savefig(fig, "learning_curve.png")

    print("  PLOT 5 — iat_per_class.png")
    iat_data = []
    for cls in cls_names:
        v = df_win.loc[df_win["label"]==cls,"mean_iat_ms"].values; v = v[v>0]
        iat_data.append(v if len(v) else np.array([0.001]))
    fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor(DARK_BG); style_ax(ax)
    bp = ax.boxplot(iat_data, patch_artist=True,
        medianprops=dict(color=TEXT_COL,lw=2),
        whiskerprops=dict(color=BORDER), capprops=dict(color=BORDER))
    for patch, col in zip(bp["boxes"], cls_palette): patch.set_facecolor(col); patch.set_alpha(0.80)
    ax.set_xticks(range(1,len(cls_names)+1))
    ax.set_xticklabels(cls_names, color=TEXT_COL, rotation=15)
    ax.set_ylabel("Mean IAT (ms)"); ax.set_title("Inter-Arrival Time per Class",color=TEXT_COL)
    ax.set_yscale("log"); plt.tight_layout(); savefig(fig,"iat_per_class.png")


# =============================================================================
# MAIN — CLI dispatch
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "IPv6 ND/RA Intrusion Detection — v18\n\n"
            "v4 CSV FALLBACK: when training on a PCAP with no in-band markers\n"
            "(i.e. captured by the v4 bash script), the pipeline automatically\n"
            "looks for <stem>_events.csv and <stem>_nodes.csv beside the PCAP.\n"
            "Override with --events-csv / --nodes-csv if files are elsewhere."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Train — v4 PCAP (CSV files auto-detected beside the PCAP):\n"
            "    python ids_pipeline_v18.py --mode train \\\n"
            "        --pcap /ipv6_research/raw_capture.pcap\n\n"
            "  # Train — explicit CSV paths:\n"
            "    python ids_pipeline_v18.py --mode train \\\n"
            "        --pcap       /ipv6_research/raw_capture.pcap \\\n"
            "        --events-csv /ipv6_research/raw_capture_events.csv \\\n"
            "        --nodes-csv  /ipv6_research/raw_capture_nodes.csv\n\n"
            "  # External validation on a feature CSV:\n"
            "    python ids_pipeline_v18.py --mode evaluate-csv \\\n"
            "        --csv Labdataset.csv --label-column Label \\\n"
            "        --label-map '{\"BENIGN\":\"Normal\"}'"
        ),
    )
    ap.add_argument("--mode", choices=["train","evaluate-pcap","evaluate-csv"],
                    required=True, help="Operating mode")
    ap.add_argument("--pcap",  default=None, help="Input PCAP file")
    ap.add_argument("--csv",   default=None, help="Input CSV file (evaluate-csv mode)")
    ap.add_argument("--model-in",  default="./ids_model_v18.joblib",
                    help="Path to load trained model bundle")
    ap.add_argument("--model-out", default="./ids_model_v18.joblib",
                    help="Path to save trained model bundle")
    # v4 CSV fallback
    ap.add_argument("--events-csv", default=None,
                    help="v4 events CSV (auto-detected from PCAP path when omitted)")
    ap.add_argument("--nodes-csv",  default=None,
                    help="v4 nodes CSV  (auto-detected from PCAP path when omitted)")
    # evaluate options
    ap.add_argument("--external-attacker-mac", default=None,
                    help="Attacker MAC for labelling external PCAPs without markers")
    ap.add_argument("--label-column", default="label",
                    help="Label column name in external CSV (default: 'label')")
    ap.add_argument("--label-map", default=None,
                    help='JSON: {"external_label":"internal_class"} e.g. \'{"BENIGN":"Normal"}\'')
    ap.add_argument("--column-map", default=None,
                    help='JSON: {"external_col":"internal_feature"} for column renaming')
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip plot generation (faster for batch runs)")
    return ap.parse_args()


def main():
    args = parse_args()
    if   args.mode == "train":
        if not args.pcap: print("[FATAL] --mode train requires --pcap"); sys.exit(1)
        run_training(args)
    elif args.mode == "evaluate-pcap":
        if not args.pcap: print("[FATAL] --mode evaluate-pcap requires --pcap"); sys.exit(1)
        run_evaluate_pcap(args)
    elif args.mode == "evaluate-csv":
        if not args.csv:  print("[FATAL] --mode evaluate-csv requires --csv");  sys.exit(1)
        run_evaluate_csv(args)


if __name__ == "__main__":
    main()