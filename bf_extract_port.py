import numpy as np
from scipy.signal import find_peaks

# ============================================================
# EXTRACTION -- faithful port of MATLAB extractPatternFeatures.m
# (grid in MATLAB: theta = -90:0.1:90)
# ============================================================
EX_PK_PROM = 0.1        # islocalmax(AF_dB,'MinProminence',0.1)
EX_NULL_PROM = 1.0      # islocalmin(AF_dB,'MinProminence',1)
EX_NULL_DEPTH = -30.0   # nullDepthTol
EX_GAP_DEG = 1.5        # ripple tolerance of the outward walk
EX_EDGE_FRAC = 0.15     # passband ripple ignores the outer 15 % on each side
EX_SPREAD_MAX = 2.0     # max spread inside a 3-peak SLL group (dB)
EX_SKIRT_MARGIN = 1.0   # skirt is finished 1 dB above the median floor
EX_GUARD_DEG = 2.0      # guard around beams when estimating the floor


def _interp_cross(y1, y2, x1, x2, level):
    return x1 + (level - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1


def _walk_out(db, i_seed, i_bound, d, level, gap_n):
    last, i, cnt = i_seed, i_seed, 0
    while True:
        nxt = i + d
        if (d > 0 and nxt > i_bound) or (d < 0 and nxt < i_bound):
            break
        i = nxt
        if db[i] >= level:
            last, cnt = i, 0
        else:
            cnt += 1
            if cnt > gap_n:
                break
    return last


def _crossings(theta, db, i_min, i_seed, i_max, level, gap_n):
    L = R = np.nan
    iL = _walk_out(db, i_seed, i_min, -1, level, gap_n)
    if iL > i_min:
        L = _interp_cross(db[iL - 1], db[iL], theta[iL - 1], theta[iL], level)
    iR = _walk_out(db, i_seed, i_max, +1, level, gap_n)
    if iR < i_max:
        R = _interp_cross(db[iR], db[iR + 1], theta[iR], theta[iR + 1], level)
    return L, R


def _extract_beams(theta, db, ang, wid):
    nB, nT = len(ang), len(theta)
    gap_n = int(round(EX_GAP_DEG / (theta[1] - theta[0])))

    seed = np.zeros(nB, int)
    for b in range(nB):
        win = np.where(np.abs(theta - ang[b]) <= wid[b] / 2.0 + 1.0)[0]
        if win.size == 0:
            win = np.array([int(np.argmin(np.abs(theta - ang[b])))])
        seed[b] = win[int(np.argmax(db[win]))]
    order = np.argsort(seed, kind="stable")

    B = [dict(angle=np.nan, gain=np.nan, bw=np.nan, ripple=np.nan,
              L=np.nan, R=np.nan, iMin=0, iMax=nT - 1, status="none") for _ in range(nB)]

    for r in range(nB):
        b = int(order[r])
        i_seed = int(seed[b])
        if r > 0:
            prev = int(seed[order[r - 1]])
            i_min = prev + int(np.argmin(db[prev:i_seed + 1]))
        else:
            i_min = 0
        if r < nB - 1:
            nxt = int(seed[order[r + 1]])
            i_max = i_seed + int(np.argmin(db[i_seed:nxt + 1]))
        else:
            i_max = nT - 1
        B[b]["iMin"], B[b]["iMax"] = i_min, i_max

        level = db[i_seed] - 3.0
        ref_gain = db[i_seed]
        L = R = np.nan
        for _ in range(30):
            L, R = _crossings(theta, db, i_min, i_seed, i_max, level, gap_n)
            if np.isnan(L) or np.isnan(R):
                break
            mask = (theta >= L) & (theta <= R)
            if not mask.any():
                L = R = np.nan
                break
            ref_gain = 10.0 * np.log10(np.mean(10.0 ** (db[mask] / 10.0)))
            new_level = ref_gain - 3.0
            if abs(new_level - level) < 1e-4:
                level = new_level
                break
            level = new_level

        if np.isnan(L) or np.isnan(R):
            B[b].update(status="merged/no crossing", angle=float(theta[i_seed]),
                        gain=float(db[i_seed]))
            continue

        mask = (theta >= L) & (theta <= R)
        bwb = R - L
        inner = (theta >= L + EX_EDGE_FRAC * bwb) & (theta <= R - EX_EDGE_FRAC * bwb)
        if not inner.any():
            inner = mask
        B[b].update(L=float(L), R=float(R), bw=float(bwb), angle=float((L + R) / 2.0),
                    gain=float(ref_gain),
                    ripple=float(db[inner].max() - db[inner].min()), status="ok")
    return B


def _scan_group(cand, peak_gain):
    for q in range(len(cand) - 2):
        t = np.asarray(cand[q:q + 3])
        g = peak_gain[t]
        if g.max() - g.min() <= EX_SPREAD_MAX:
            j = int(np.argmax(g))
            return True, float(g[j]), int(t[j])
    return False, np.nan, -1


def _extract_sll(theta, db, pk_idx, B):
    peak_loc, peak_gain = theta[pk_idx], db[pk_idx]
    nB, nP, nT = len(B), len(pk_idx), len(theta)

    Lb = np.zeros(nB); Rb = np.zeros(nB)
    for b in range(nB):
        if np.isnan(B[b]["L"]) or np.isnan(B[b]["R"]):
            Lb[b], Rb[b] = theta[B[b]["iMin"]], theta[B[b]["iMax"]]
        else:
            Lb[b], Rb[b] = B[b]["L"], B[b]["R"]

    near = np.zeros(nT, bool)
    for b in range(nB):
        near |= (theta >= Lb[b] - EX_GUARD_DEG) & (theta <= Rb[b] + EX_GUARD_DEG)
    v = np.sort(db[~near])
    if v.size == 0:
        floor_level = db.min()
    else:
        floor_level = v[max(1, int(np.floor(0.5 * v.size + 0.5))) - 1]
    skirt = floor_level + EX_SKIRT_MARGIN

    LbS, RbS = Lb.copy(), Rb.copy()
    for b in range(nB):
        iL = max(int(np.searchsorted(theta, Lb[b], side="right")) - 1, 0)
        k = np.where(db[:iL + 1] <= skirt)[0]
        LbS[b] = theta[0] if k.size == 0 else theta[k[-1]]
        iR = min(int(np.searchsorted(theta, Rb[b], side="left")), nT - 1)
        k = np.where(db[iR:] <= skirt)[0]
        RbS[b] = theta[-1] if k.size == 0 else theta[iR + k[0]]

    region = np.zeros(nP, bool)
    for b in range(nB):
        region |= (peak_loc >= LbS[b]) & (peak_loc <= RbS[b])

    cands = []
    for b in range(nB):
        left = np.where(peak_loc < LbS[b])[0]
        if left.size:
            iL = int(left[-1])
            reg = np.where(region[:iL + 1])[0]
            stop = int(reg[-1]) if reg.size else -1
            ok, g, idx = _scan_group(list(range(iL, stop, -1)), peak_gain)
            if ok:
                cands.append((g, float(peak_loc[idx])))
        right = np.where(peak_loc > RbS[b])[0]
        if right.size:
            iR = int(right[0])
            reg = np.where(region[iR:])[0]
            last = nP - 1 if reg.size == 0 else iR + int(reg[0]) - 1
            ok, g, idx = _scan_group(list(range(iR, last + 1)), peak_gain)
            if ok:
                cands.append((g, float(peak_loc[idx])))
    if not cands:
        return np.nan, np.nan
    return max(cands, key=lambda t: t[0])


def extract_features_matlab(db, theta, beam_angles, beam_widths):
    """db: pattern in dB normalised to its GLOBAL max; beam_angles / beam_widths: the (extracted) spec
    values, used only to seed the beam search (MATLAB seeds with the requested spec)."""
    pk_idx, _ = find_peaks(db, prominence=EX_PK_PROM)
    B = _extract_beams(theta, db, np.asarray(beam_angles, float), np.asarray(beam_widths, float))
    sll, sll_ang = _extract_sll(theta, db, pk_idx, B)
    nl, _ = find_peaks(-db, prominence=EX_NULL_PROM)
    nl = nl[db[nl] <= EX_NULL_DEPTH]
    return dict(null_angles=theta[nl], null_depth=db[nl], beams=B, sll=sll, sll_angle=sll_ang)