# =============================================================================
# ML-Based IDS for IPv6 ND & RA Attack Detection — Digital Twin (Containerlab)
# Master's Thesis Project
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"   # Windows/Anaconda memory fragmentation fix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go

from scapy.all import rdpcap, IPv6, ICMPv6ND_RA, ICMPv6ND_NS, ICMPv6ND_NA
from scapy.all import ICMPv6EchoRequest, ICMPv6EchoReply, Ether, UDP, TCP
from scapy.layers.inet6 import ICMPv6NDOptPrefixInfo, ICMPv6NDOptSrcLLAddr

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PCAP_PATH        = "./raw_capture.pcap"
CSV_PATH         = "./ipv6_ids_dataset.csv"
RANDOM_STATE     = 42
N_FOLDS          = 10
TEST_SIZE        = 0.20
MAX_K_FEATURES   = 20          # plot accuracy up to this many features
REPORT_TOP_K     = 5           # report metrics for top-5 features

# Burst-detection thresholds (packets per second per source)
RA_BURST_PPS  = 5    # >= 5 RA/s from one source  → RA attack
NS_BURST_PPS  = 10   # >= 10 NS/s from one source → ND attack
WINDOW_SEC    = 1.0  # sliding-window width in seconds

# ─────────────────────────────────────────────────────────────────────────────
# SHARED RF FACTORY — single definition used everywhere
# ─────────────────────────────────────────────────────────────────────────────
def make_rf():
    return RandomForestClassifier(
        n_estimators=100,        # reduced from 200/300 to limit memory
        max_depth=15,            # capped depth keeps each tree lean
        max_features="sqrt",     # default, further reduces per-tree allocation
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1                 # KEY FIX: sequential on Windows avoids heap fragmentation
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PCAP LOADING & FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — Loading PCAP and extracting features ...")
print("=" * 70)

packets = rdpcap(PCAP_PATH)
print(f"  Total packets in capture : {len(packets)}")

# ── helpers ──────────────────────────────────────────────────────────────────
def ipv6_scope(addr: str) -> int:
    """Encode IPv6 address scope as integer."""
    if addr.startswith("fe80"):   return 1   # link-local
    if addr.startswith("ff"):     return 2   # multicast
    if addr == "::1":             return 3   # loopback
    return 0                                 # global unicast / other

def prefix_len_from_pkt(pkt) -> int:
    """Extract RA prefix length if present."""
    try:
        return int(pkt[ICMPv6NDOptPrefixInfo].prefixlen)
    except Exception:
        return 0

def has_src_lladdr(pkt) -> int:
    try:
        _ = pkt[ICMPv6NDOptSrcLLAddr]
        return 1
    except Exception:
        return 0

# ── first pass: compute per-source RA/NS rates for burst labelling ────────────
ra_times  = defaultdict(list)   # src → [timestamps]
ns_times  = defaultdict(list)

for pkt in packets:
    if not pkt.haslayer(IPv6):
        continue
    ts  = float(pkt.time)
    src = pkt[IPv6].src
    if pkt.haslayer(ICMPv6ND_RA):
        ra_times[src].append(ts)
    elif pkt.haslayer(ICMPv6ND_NS):
        ns_times[src].append(ts)

def max_burst_rate(times: list, window: float) -> float:
    """Return maximum packets-per-second in any sliding window."""
    if len(times) < 2:
        return 0.0
    times = sorted(times)
    max_r, lo = 0.0, 0
    for hi in range(len(times)):
        while times[hi] - times[lo] > window:
            lo += 1
        rate = (hi - lo + 1) / window
        if rate > max_r:
            max_r = rate
    return max_r

ra_burst  = {src: max_burst_rate(ts, WINDOW_SEC) for src, ts in ra_times.items()}
ns_burst  = {src: max_burst_rate(ts, WINDOW_SEC) for src, ts in ns_times.items()}

attack_ra_srcs = {s for s, r in ra_burst.items() if r >= RA_BURST_PPS}
attack_ns_srcs = {s for s, r in ns_burst.items() if r >= NS_BURST_PPS}

print(f"  Detected RA-attack source(s)  : {attack_ra_srcs or 'none (threshold not met)'}")
print(f"  Detected ND-attack source(s)  : {attack_ns_srcs or 'none (threshold not met)'}")

# Fallback: if burst detection finds nothing, label by ICMPv6 type only
fallback = len(attack_ra_srcs) == 0 and len(attack_ns_srcs) == 0
if fallback:
    print("  [WARN] Burst thresholds not met — using ICMPv6-type-only labelling.")

# ── second pass: extract features per packet ──────────────────────────────────
records = []

for pkt in packets:
    if not pkt.haslayer(IPv6):
        continue

    ip6   = pkt[IPv6]
    ts    = float(pkt.time)
    src   = ip6.src
    dst   = ip6.dst

    # ── layer-3 features ──────────────────────────────────────────────────
    pkt_len       = len(pkt)
    ip6_plen      = int(ip6.plen)
    hop_limit     = int(ip6.hlim)
    next_hdr      = int(ip6.nh)
    src_scope     = ipv6_scope(src)
    dst_scope     = ipv6_scope(dst)
    is_multicast  = int(dst.startswith("ff"))
    is_link_local = int(src.startswith("fe80"))

    # ── ICMPv6 / ND features ──────────────────────────────────────────────
    icmpv6_type   = 0
    icmpv6_code   = 0
    is_ra         = 0
    is_ns         = 0
    is_na         = 0
    is_echo       = 0
    ra_managed    = 0
    ra_other      = 0
    ra_lifetime   = 0
    ra_reachtime  = 0
    ra_retranstimer = 0
    ra_prefix_len = 0
    has_slla      = 0
    nd_target_scope = 0

    if pkt.haslayer("ICMPv6"):
        try:
            icmpv6_type = int(pkt["ICMPv6"].type)
            icmpv6_code = int(pkt["ICMPv6"].code)
        except Exception:
            pass

    if pkt.haslayer(ICMPv6ND_RA):
        is_ra          = 1
        ra             = pkt[ICMPv6ND_RA]
        ra_managed     = int(getattr(ra, "M", 0))
        ra_other       = int(getattr(ra, "O", 0))
        ra_lifetime    = int(getattr(ra, "routerlifetime", 0))
        ra_reachtime   = int(getattr(ra, "reachabletime", 0))
        ra_retranstimer= int(getattr(ra, "retranstimer", 0))
        ra_prefix_len  = prefix_len_from_pkt(pkt)
        has_slla       = has_src_lladdr(pkt)

    elif pkt.haslayer(ICMPv6ND_NS):
        is_ns           = 1
        tgt             = str(getattr(pkt[ICMPv6ND_NS], "tgt", "::"))
        nd_target_scope = ipv6_scope(tgt)
        has_slla        = has_src_lladdr(pkt)

    elif pkt.haslayer(ICMPv6ND_NA):
        is_na           = 1
        tgt             = str(getattr(pkt[ICMPv6ND_NA], "tgt", "::"))
        nd_target_scope = ipv6_scope(tgt)

    elif pkt.haslayer(ICMPv6EchoRequest) or pkt.haslayer(ICMPv6EchoReply):
        is_echo = 1

    # ── transport features ────────────────────────────────────────────────
    has_udp  = int(pkt.haslayer(UDP))
    has_tcp  = int(pkt.haslayer(TCP))
    src_port = int(pkt[UDP].sport if has_udp else (pkt[TCP].sport if has_tcp else 0))
    dst_port = int(pkt[UDP].dport if has_udp else (pkt[TCP].dport if has_tcp else 0))

    # ── label ─────────────────────────────────────────────────────────────
    if fallback:
        if is_ra:
            label = "RA_Attack"
        elif is_ns:
            label = "ND_Attack"
        else:
            label = "Normal"
    else:
        if is_ra and src in attack_ra_srcs:
            label = "RA_Attack"
        elif is_ns and src in attack_ns_srcs:
            label = "ND_Attack"
        else:
            label = "Normal"

    records.append({
        "timestamp"        : ts,
        "pkt_len"          : pkt_len,
        "ip6_plen"         : ip6_plen,
        "hop_limit"        : hop_limit,
        "next_hdr"         : next_hdr,
        "src_scope"        : src_scope,
        "dst_scope"        : dst_scope,
        "is_multicast"     : is_multicast,
        "is_link_local"    : is_link_local,
        "icmpv6_type"      : icmpv6_type,
        "icmpv6_code"      : icmpv6_code,
        "is_ra"            : is_ra,
        "is_ns"            : is_ns,
        "is_na"            : is_na,
        "is_echo"          : is_echo,
        "ra_managed"       : ra_managed,
        "ra_other"         : ra_other,
        "ra_lifetime"      : ra_lifetime,
        "ra_reachtime"     : ra_reachtime,
        "ra_retranstimer"  : ra_retranstimer,
        "ra_prefix_len"    : ra_prefix_len,
        "has_slla"         : has_slla,
        "nd_target_scope"  : nd_target_scope,
        "has_udp"          : has_udp,
        "has_tcp"          : has_tcp,
        "src_port"         : src_port,
        "dst_port"         : dst_port,
        "label"            : label,
    })

df = pd.DataFrame(records)

print(f"\n  DataFrame shape : {df.shape}")
print("\n  Class distribution:")
print(df["label"].value_counts().to_string())

df.to_csv(CSV_PATH, index=False)
print(f"\n  Dataset saved -> {CSV_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING & FEATURE SELECTION (SelectKBest)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — Preprocessing & SelectKBest feature ranking ...")
print("=" * 70)

FEATURE_COLS = [c for c in df.columns if c not in ("timestamp", "label")]
X_raw = df[FEATURE_COLS].values.astype(float)
le    = LabelEncoder()
y     = le.fit_transform(df["label"])

print(f"  Classes           : {list(le.classes_)}")
print(f"  Feature pool size : {len(FEATURE_COLS)}")

# Fit SelectKBest on ALL features to get importances / ranking
k_all     = min(MAX_K_FEATURES, len(FEATURE_COLS))
selector  = SelectKBest(f_classif, k=k_all)
selector.fit(X_raw, y)

scores_df = pd.DataFrame({
    "feature" : FEATURE_COLS,
    "score"   : selector.scores_,
}).sort_values("score", ascending=False).reset_index(drop=True)

print("\n  Feature importance ranking (SelectKBest / ANOVA-F):")
print(scores_df.to_string(index=False))

ranked_features = scores_df["feature"].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT  +  10-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — 80/20 split  +  10-fold CV across increasing K features ...")
print("=" * 70)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

k_values      = list(range(1, k_all + 1))
cv_acc_list   = []   # mean CV accuracy for each K
test_acc_list = []   # held-out test accuracy for each K

for k in k_values:
    top_k_feats = ranked_features[:k]
    idx         = [FEATURE_COLS.index(f) for f in top_k_feats]

    X_tr_k = X_train_full[:, idx]
    X_te_k = X_test_full[:, idx]

    # 10-fold CV on training portion
    fold_accs = []
    for tr_idx, val_idx in skf.split(X_tr_k, y_train):
        clf = make_rf()
        clf.fit(X_tr_k[tr_idx], y_train[tr_idx])
        pred_val = clf.predict(X_tr_k[val_idx])
        fold_accs.append(accuracy_score(y_train[val_idx], pred_val))

    cv_acc_list.append(np.mean(fold_accs))

    # Retrain on full training set, evaluate on held-out test set
    clf_full = make_rf()
    clf_full.fit(X_tr_k, y_train)
    pred_test = clf_full.predict(X_te_k)
    test_acc_list.append(accuracy_score(y_test, pred_test))

    print(f"  k={k:2d}  CV acc={cv_acc_list[-1]:.4f}   Test acc={test_acc_list[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DETAILED METRICS FOR TOP-5 FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — Detailed metrics for top-5 features ...")
print("=" * 70)

top5     = ranked_features[:REPORT_TOP_K]
idx5     = [FEATURE_COLS.index(f) for f in top5]
X_tr5    = X_train_full[:, idx5]
X_te5    = X_test_full[:, idx5]

clf5 = make_rf()
clf5.fit(X_tr5, y_train)
pred5 = clf5.predict(X_te5)

acc5  = accuracy_score(y_test, pred5)
bacc5 = balanced_accuracy_score(y_test, pred5)
mf1_5 = f1_score(y_test, pred5, average="macro")

print(f"\n  Top-5 features used  : {top5}")
print(f"\n  Test Accuracy        : {acc5:.4f}  ({acc5*100:.2f}%)")
print(f"  Balanced Accuracy    : {bacc5:.4f}  ({bacc5*100:.2f}%)")
print(f"  Macro F1-Score       : {mf1_5:.4f}  ({mf1_5*100:.2f}%)")
print("\n  Full Classification Report (top-5 features):")
print(classification_report(y_test, pred5, target_names=le.classes_))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN FINAL MODEL ON BEST-K FEATURES  (used for visualisations)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — Training final model on best-K ranked features ...")
print("=" * 70)

best_k     = int(np.argmax(test_acc_list)) + 1
best_feats = ranked_features[:best_k]
best_idx   = [FEATURE_COLS.index(f) for f in best_feats]

X_tr_best = X_train_full[:, best_idx]
X_te_best = X_test_full[:, best_idx]

clf_final = make_rf()
clf_final.fit(X_tr_best, y_train)
pred_final = clf_final.predict(X_te_best)

acc_final  = accuracy_score(y_test, pred_final)
bacc_final = balanced_accuracy_score(y_test, pred_final)
mf1_final  = f1_score(y_test, pred_final, average="macro")

print(f"\n  Best K (by test accuracy) : {best_k}")
print(f"  Best features             : {best_feats}")
print(f"\n  Final Test Accuracy       : {acc_final:.4f}  ({acc_final*100:.2f}%)")
print(f"  Final Balanced Accuracy   : {bacc_final:.4f}  ({bacc_final*100:.2f}%)")
print(f"  Final Macro F1-Score      : {mf1_final:.4f}  ({mf1_final*100:.2f}%)")
print("\n  Full Classification Report (final model):")
print(classification_report(y_test, pred_final, target_names=le.classes_))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT — Test Accuracy vs Number of Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — Accuracy vs Number of Features (plot) ...")
print("=" * 70)

fig_acc, ax_acc = plt.subplots(figsize=(12, 5))

ax_acc.plot(k_values, [v * 100 for v in test_acc_list],
            marker="o", linewidth=2, color="#1f77b4", label="Test Accuracy")
ax_acc.plot(k_values, [v * 100 for v in cv_acc_list],
            marker="s", linewidth=2, linestyle="--", color="#ff7f0e",
            label="CV Accuracy (10-fold mean)")

ax_acc.axvline(x=best_k, color="green", linestyle=":", linewidth=1.5,
               label=f"Best K = {best_k}")
ax_acc.axhline(y=75, color="grey", linestyle="--", linewidth=1, alpha=0.6,
               label="75% threshold")
ax_acc.axhline(y=80, color="grey", linestyle="--", linewidth=1, alpha=0.6,
               label="80% threshold")

ax_acc.fill_between(k_values,
                    [v * 100 for v in test_acc_list],
                    [v * 100 for v in cv_acc_list],
                    alpha=0.08, color="#1f77b4")

ax_acc.set_xlabel("Number of Features (K)", fontsize=13)
ax_acc.set_ylabel("Accuracy (%)", fontsize=13)
ax_acc.set_title("Random Forest — Test & CV Accuracy vs Number of Features\n"
                 "IPv6 ND/RA Attack Detection (Digital Twin — Containerlab)",
                 fontsize=13, fontweight="bold")
ax_acc.set_xticks(k_values)
ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax_acc.legend(fontsize=10)
ax_acc.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_features.png", dpi=150)
plt.show()
print("  Saved -> accuracy_vs_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PREDICTION VISUALISATIONS
#     7a. Single comparative LINE graph
#     7b. Single comparative BAR graph
#     7c. Interactive 3D Plotly graph
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — Prediction visualisations ...")
print("=" * 70)

classes   = list(le.classes_)
n_classes = len(classes)

# Actual vs predicted counts
actual_counts = {cls: int(np.sum(y_test == le.transform([cls])[0])) for cls in classes}
pred_counts   = {cls: int(np.sum(pred_final == le.transform([cls])[0])) for cls in classes}

# Per-class metrics
per_class_acc = {}
per_class_f1  = {}
for cls in classes:
    idx_cls = le.transform([cls])[0]
    mask    = (y_test == idx_cls)
    if mask.sum() > 0:
        per_class_acc[cls] = float(np.sum((pred_final == idx_cls) & mask) / mask.sum())
        per_class_f1[cls]  = float(f1_score(y_test, pred_final,
                                             labels=[idx_cls], average="macro"))
    else:
        per_class_acc[cls] = 0.0
        per_class_f1[cls]  = 0.0

print("\n  Per-class detection accuracy:")
for cls in classes:
    print(f"    {cls:12s} : {per_class_acc[cls]*100:.2f}%  |  "
          f"Actual={actual_counts[cls]}  Predicted={pred_counts[cls]}")

# ── colour palette ─────────────────────────────────────────────────────────
COLORS = {"ND_Attack": "#e74c3c", "Normal": "#2ecc71", "RA_Attack": "#3498db"}
BAR_W  = 0.35
x_pos  = np.arange(n_classes)

# ─────────────────────────────────────────
# 7a.  COMPARATIVE LINE GRAPH
# ─────────────────────────────────────────
actual_vals  = [actual_counts[c] for c in classes]
predict_vals = [pred_counts[c]   for c in classes]

fig_line, ax_line = plt.subplots(figsize=(10, 5))

ax_line.plot(classes, actual_vals,  marker="o", linewidth=2.5,
             color="#2c3e50", label="Actual Count",    markersize=9)
ax_line.plot(classes, predict_vals, marker="D", linewidth=2.5,
             linestyle="--", color="#e74c3c",
             label="Predicted Count", markersize=9)

for i, cls in enumerate(classes):
    ax_line.annotate(str(actual_vals[i]),
                     (cls, actual_vals[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, color="#2c3e50")
    ax_line.annotate(str(predict_vals[i]),
                     (cls, predict_vals[i]), textcoords="offset points",
                     xytext=(0, -18), ha="center", fontsize=10, color="#e74c3c")

ax_line.fill_between(classes, actual_vals, predict_vals,
                     alpha=0.10, color="#9b59b6")
ax_line.set_xlabel("Traffic Class", fontsize=13)
ax_line.set_ylabel("Number of Packets (Test Set)", fontsize=13)
ax_line.set_title("IPv6 IDS — Actual vs Predicted Class Distribution\n"
                  "(Comparative Line Graph)", fontsize=13, fontweight="bold")
ax_line.legend(fontsize=11)
ax_line.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("line_actual_vs_predicted.png", dpi=150)
plt.show()
print("  Saved -> line_actual_vs_predicted.png")

# ─────────────────────────────────────────
# 7b.  COMPARATIVE BAR GRAPH
# ─────────────────────────────────────────
fig_bar, axes_bar = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Actual vs Predicted counts
bars1 = axes_bar[0].bar(x_pos - BAR_W / 2, actual_vals,  BAR_W,
                         label="Actual",    color="#2c3e50", alpha=0.85, edgecolor="white")
bars2 = axes_bar[0].bar(x_pos + BAR_W / 2, predict_vals, BAR_W,
                         label="Predicted", color="#e74c3c", alpha=0.85, edgecolor="white")

for bar in bars1:
    axes_bar[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(actual_vals) * 0.01,
                     str(int(bar.get_height())),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar in bars2:
    axes_bar[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(actual_vals) * 0.01,
                     str(int(bar.get_height())),
                     ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color="#e74c3c")

axes_bar[0].set_xticks(x_pos)
axes_bar[0].set_xticklabels(classes, fontsize=11)
axes_bar[0].set_xlabel("Traffic Class", fontsize=12)
axes_bar[0].set_ylabel("Number of Packets", fontsize=12)
axes_bar[0].set_title("Actual vs Predicted\nPacket Counts", fontsize=12, fontweight="bold")
axes_bar[0].legend(fontsize=10)
axes_bar[0].grid(True, axis="y", alpha=0.25)

# Right panel: Per-class detection accuracy
acc_vals    = [per_class_acc[c] * 100 for c in classes]
bar_colors  = [COLORS[c] for c in classes]

bars3 = axes_bar[1].bar(x_pos, acc_vals, 0.5,
                         color=bar_colors, alpha=0.85, edgecolor="white")
for bar, val in zip(bars3, acc_vals):
    axes_bar[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

axes_bar[1].axhline(y=75, color="grey", linestyle="--", linewidth=1, alpha=0.7,
                     label="75% threshold")
axes_bar[1].axhline(y=80, color="grey", linestyle="-.", linewidth=1, alpha=0.7,
                     label="80% threshold")
axes_bar[1].set_xticks(x_pos)
axes_bar[1].set_xticklabels(classes, fontsize=11)
axes_bar[1].set_xlabel("Traffic Class", fontsize=12)
axes_bar[1].set_ylabel("Detection Accuracy (%)", fontsize=12)
axes_bar[1].set_title("Per-Class Detection Accuracy\n(Random Forest)",
                       fontsize=12, fontweight="bold")
axes_bar[1].set_ylim(0, 110)
axes_bar[1].legend(fontsize=9)
axes_bar[1].grid(True, axis="y", alpha=0.25)

fig_bar.suptitle("IPv6 ND/RA Attack IDS — Random Forest Comparative Bar Graph",
                 fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("bar_comparative.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> bar_comparative.png")

# ─────────────────────────────────────────
# 7c.  INTERACTIVE 3-D PLOTLY GRAPH
# ─────────────────────────────────────────
feat3d_names = ranked_features[:3]
feat3d_idx   = [FEATURE_COLS.index(f) for f in feat3d_names]

X_test_3d   = X_test_full[:, feat3d_idx]
pred_labels = le.inverse_transform(pred_final)

plotly_colors = {
    "ND_Attack": "#e74c3c",
    "Normal"   : "#2ecc71",
    "RA_Attack": "#3498db"
}

fig3d = go.Figure()

for cls in classes:
    mask_cls = pred_labels == cls
    fig3d.add_trace(go.Scatter3d(
        x=X_test_3d[mask_cls, 0],
        y=X_test_3d[mask_cls, 1],
        z=X_test_3d[mask_cls, 2],
        mode="markers",
        name=cls,
        marker=dict(
            size=3,
            color=plotly_colors[cls],
            opacity=0.75,
            line=dict(width=0)
        ),
        hovertemplate=(
            f"<b>{cls}</b><br>"
            f"{feat3d_names[0]}: %{{x}}<br>"
            f"{feat3d_names[1]}: %{{y}}<br>"
            f"{feat3d_names[2]}: %{{z}}<extra></extra>"
        )
    ))

fig3d.update_layout(
    title=dict(
        text=(
            "IPv6 IDS — Predicted Traffic Classification (Interactive 3-D)<br>"
            "<sup>Digital Twin · Containerlab · Random Forest</sup>"
        ),
        x=0.5, xanchor="center"
    ),
    scene=dict(
        xaxis_title=feat3d_names[0],
        yaxis_title=feat3d_names[1],
        zaxis_title=feat3d_names[2],
        xaxis=dict(backgroundcolor="rgba(240,240,255,0.8)", gridcolor="white"),
        yaxis=dict(backgroundcolor="rgba(240,255,240,0.8)", gridcolor="white"),
        zaxis=dict(backgroundcolor="rgba(255,240,240,0.8)", gridcolor="white"),
    ),
    legend=dict(
        title="Predicted Class",
        itemsizing="constant",
        font=dict(size=13)
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    width=950, height=700
)

fig3d.write_html("3d_interactive_predictions.html")
fig3d.show()
print("  Saved -> 3d_interactive_predictions.html  (open in any browser)")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FINAL SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY REPORT")
print("=" * 70)
print(f"\n  PCAP file          : {PCAP_PATH}")
print(f"  Total IPv6 packets : {len(df)}")
print(f"  Train / Test split : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
print(f"  CV folds           : {N_FOLDS}")
print(f"  Classifier         : Random Forest (100 trees, max_depth=15, n_jobs=1)")
print(f"\n  -- TOP-5 FEATURES --------------------------------------------------")
print(f"  Features           : {top5}")
print(f"  Test Accuracy      : {acc5*100:.2f}%")
print(f"  Balanced Accuracy  : {bacc5*100:.2f}%")
print(f"  Macro F1           : {mf1_5*100:.2f}%")
print(f"\n  -- BEST-K MODEL  (K = {best_k}) ------------------------------------")
print(f"  Features           : {best_feats}")
print(f"  Test Accuracy      : {acc_final*100:.2f}%")
print(f"  Balanced Accuracy  : {bacc_final*100:.2f}%")
print(f"  Macro F1           : {mf1_final*100:.2f}%")
print(f"\n  Output files generated:")
print(f"    ipv6_ids_dataset.csv")
print(f"    accuracy_vs_features.png")
print(f"    line_actual_vs_predicted.png")
print(f"    bar_comparative.png")
print(f"    3d_interactive_predictions.html")
print("=" * 70)