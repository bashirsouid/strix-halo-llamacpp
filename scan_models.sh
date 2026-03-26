#!/usr/bin/env bash
# scan_models.sh — find all GGUFs across every search directory.
# Identifies duplicates wasting disk space.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/config.env"

echo -e "${BOLD}"
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════╗
║      Model Scanner — all GGUFs across search directories         ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

_build_search_dirs

echo -e "${BOLD}Active search directories (${#_SEARCH_DIRS[@]}):${NC}"
for d in "${_SEARCH_DIRS[@]}"; do
  echo "  ${CYN}${d}${NC}"
done
echo ""

echo -e "${BOLD}── Scanning... ──────────────────────────────────────────────────${NC}"

declare -a all_files=()
for dir in "${_SEARCH_DIRS[@]}"; do
  while IFS= read -r -d '' f; do
    all_files+=("$f")
  done < <(find "$dir" -maxdepth 10 -name "*.gguf" -print0 2>/dev/null)
done

if (( ${#all_files[@]} == 0 )); then
  _lib_warn "No .gguf files found in any search directory."
  exit 0
fi

echo ""
echo -e "${BOLD}── All GGUFs found ──────────────────────────────────────────────${NC}"
printf "%-8s  %-14s  %s\n" "SIZE" "QUANT" "PATH"
echo "────────────────────────────────────────────────────────────────────"

declare -A real_to_first=()
declare -a dup_pairs=()

for f in "${all_files[@]}"; do
  rp=$(readlink -f "$f" 2>/dev/null || echo "$f")
  size=$(du -sh "$f" 2>/dev/null | cut -f1 || echo "?")
  quant=$(basename "$f" \
    | grep -oP '(?i)(IQ[0-9]_[A-Z]+|Q[0-9]_K_[A-Z]+|Q[0-9]_[0-9]+|F16|F32|BF16)' \
    | head -1 || echo "?")

  if [[ -n "${real_to_first[$rp]+_}" ]]; then
    printf "${YLW}%-8s${NC}  %-14s  ${YLW}%s${NC}  ${DIM}(dup)${NC}\n" \
      "$size" "$quant" "$f"
    dup_pairs+=("${real_to_first[$rp]}||${f}")
  else
    real_to_first[$rp]="$f"
    printf "${GRN}%-8s${NC}  %-14s  %s\n" "$size" "$quant" "$f"
  fi
done

echo ""
TOTAL=$(du -shc "${all_files[@]}" 2>/dev/null | tail -1 | cut -f1 || echo "?")
echo -e "  Total on disk : ${BOLD}${TOTAL}${NC}  (${#all_files[@]} files)"

if (( ${#dup_pairs[@]} > 0 )); then
  echo ""
  echo -e "${BOLD}── Duplicates (same inode/content, multiple paths) ─────────────${NC}"
  _lib_warn "${#dup_pairs[@]} duplicate(s) — safe to remove the highlighted paths:"
  echo ""
  for pair in "${dup_pairs[@]}"; do
    IFS='||' read -r orig dup <<< "$pair"
    dup_size=$(du -sh "$dup" 2>/dev/null | cut -f1 || echo "?")
    echo -e "  ${GRN}KEEP:${NC} $orig"
    echo -e "  ${YLW}DUP: ${NC} $dup  ${DIM}(${dup_size} recoverable)${NC}"
    echo -e "  ${DIM}  rm \"${dup}\"${NC}"
    echo ""
  done
else
  _lib_ok "No duplicate GGUFs detected."
fi
