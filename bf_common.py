"""
bf_common.py -- shared pieces used by make_prior_dataset.py and train_student.py

  * constants / pass criteria
  * checker  (faithful port of MATLAB extractPatternFeatures.m)
  * TF pattern + reference-free spec loss
  * analytic sector prior (sinc x Kaiser x geometric carrier) + exact null projection
  * student network (learned per-beam envelopes x geometric carrier)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

# ============================================================
# CONSTANTS
# ============================================================
MAX_N, MAX_B, MAX_NUL = 128, 4, 3
INPUT_DIM = 19
GAIN_SCALE = 10.0               # stored gain input = relative gain (dB, 0 = strongest beam) / GAIN_SCALE
DB_FLOOR = 1e-2
THETA_STEP_EVAL = 0.1           # checker grid (same as MATLAB: -90:0.1:90)
THETA_STEP_LOSS = 0.125         # loss / prior grid

# prior parameters per N: (kappa, kaiser_offset).  Values found by tune_prior_per_n in v6.
PRIOR_PARAMS = {32: (0.3, 11.0), 64: (0.3, 11.0), 128: (0.2, 5.0)}

# network
WIDTH, DROPOUT, WEIGHT_DECAY, GRAD_CLIP = 512, 0.1, 1e-4, 5.0

# spec-loss settings
N_WEIGHTS = {32: 1.0, 64: 1.5, 128: 2.0}
TRANS_DEG = 5.0
SLL_TOL = 0.0
BW_WIN_MARGIN = 4.0
BW_MARGIN = 0.05
NULL_PROJ_REG = 1e-4
PRIOR_PLATEAU_NORM = True

# pass criteria
PASS_GAIN_DB = 1.0
PASS_BW_REL = 0.25
PASS_ANG_REL = 0.5
PASS_SLL_DB = 2.0
PASS_NULL_DEG = 1.0
BW_TOL_FRAC = PASS_BW_REL

# extraction constants (MATLAB port)
EX_PK_PROM = 0.1
EX_NULL_PROM = 1.0
EX_NULL_DEPTH = -30.0
EX_GAP_DEG = 1.5
EX_EDGE_FRAC = 0.15
EX_SPREAD_MAX = 2.0
EX_SKIRT_MARGIN = 1.0
EX_GUARD_DEG = 2.0

BW_TAU_VAR = tf.Variable(3.0, trainable=False, dtype=tf.float32)


# ============================================================
# INPUT VECTOR LAYOUT (19 numbers)
#   0: N/128 | 1: nBeams/4 | 2: nNulls/3 | 3:7 beam angle/90 | 7:11 beam width/20
#   11:15 relative gain/GAIN_SCALE | 15:18 null angle/90 | 18: SLL/20
# ============================================================
def _pad(x, n):
    out = np.zeros(n, np.float32)
    x = np.atleast_1d(np.asarray(x, np.float64)).ravel()[:n]
    out[:len(x)] = x
    return out


def build_x(N, nb, nn, angles, widths, gain_rel, null_angles, sll):
    return np.concatenate([
        [N / MAX_N], [nb / MAX_B], [nn / MAX_NUL],
        _pad(angles, MAX_B) / 90.0,
        _pad(widths, MAX_B) / 20.0,
        _pad(gain_rel, MAX_B) / GAIN_SCALE,
        _pad(null_angles, MAX_NUL) / 90.0,
        [float(sll) / 20.0],
    ]).astype(np.float32)


def mirror_x(Xs):
    """theta -> -theta  (angles negate and reverse order; widths / gains just reverse)."""
    Xm = Xs.copy()
    nb_ = np.rint(Xs[:, 1] * MAX_B).astype(int)
    nn_ = np.rint(Xs[:, 2] * MAX_NUL).astype(int)
    for k in range(len(Xs)):
        b, m = nb_[k], nn_[k]
        Xm[k, 3:3 + b] = -Xs[k, 3:3 + b][::-1]
        Xm[k, 7:7 + b] = Xs[k, 7:7 + b][::-1]
        Xm[k, 11:11 + b] = Xs[k, 11:11 + b][::-1]
        Xm[k, 15:15 + m] = -Xs[k, 15:15 + m][::-1]
    return Xm


# ============================================================
# CHECKER: faithful port of MATLAB extractPatternFeatures.m
# ============================================================
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
    pk_idx, _ = find_peaks(db, prominence=EX_PK_PROM)
    B = _extract_beams(theta, db, np.asarray(beam_angles, float), np.asarray(beam_widths, float))
    sll, sll_ang = _extract_sll(theta, db, pk_idx, B)
    nl, _ = find_peaks(-db, prominence=EX_NULL_PROM)
    nl = nl[db[nl] <= EX_NULL_DEPTH]
    return dict(null_angles=theta[nl], null_depth=db[nl], beams=B, sll=sll, sll_angle=sll_ang)


# ============================================================
# STEERING MATRICES
# ============================================================
theta_np = np.arange(-90, 90.001, THETA_STEP_EVAL)
A_np = np.exp(1j * np.pi * np.outer(np.arange(MAX_N), np.sin(np.deg2rad(theta_np)))).astype(np.complex64)

theta_loss_np = np.arange(-90, 90.001, THETA_STEP_LOSS)
_phase_loss = np.pi * np.outer(np.arange(MAX_N), np.sin(np.deg2rad(theta_loss_np)))
A_re = tf.constant(np.cos(_phase_loss), dtype=tf.float32)
A_im = tf.constant(np.sin(_phase_loss), dtype=tf.float32)
theta_deg = tf.constant(theta_loss_np, tf.float32)[None, None, :]


def db_np(w):
    af = np.abs(w @ A_np)
    af = af / (af.max(axis=1, keepdims=True) + 1e-9)
    return 20 * np.log10(np.maximum(af, DB_FLOOR))


def db_raw(w):
    af = np.abs(w @ A_np)
    af = af / (af.max() + 1e-12)
    return 20 * np.log10(af + 1e-12)


def wsim(a, b):
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


# ============================================================
# EXACT EVALUATION (checker)
# ============================================================
def spec_report(w, x):
    nb = int(round(x[1] * MAX_B)); nn = int(round(x[2] * MAX_NUL))
    ang, wid, gain = x[3:7] * 90, x[7:11] * 20, x[11:15] * GAIN_SCALE
    nang, sll = x[15:18] * 90, x[18] * 20

    f = extract_features_matlab(db_raw(w), theta_np, ang[:nb], wid[:nb])
    gains_abs = np.array([bm["gain"] for bm in f["beams"]], dtype=float)
    gmax = np.nanmax(gains_abs) if np.isfinite(gains_abs).any() else np.nan

    g_ok_l, b_ok_l, a_ok_l = [], [], []
    g_err, b_err, a_err = [], [], []
    for b in range(nb):
        bm = f["beams"][b]
        ge = abs((bm["gain"] - gmax) - gain[b]) if np.isfinite(bm["gain"]) else np.nan
        be = abs(bm["bw"] - wid[b]) if np.isfinite(bm["bw"]) else np.nan
        ae = abs(bm["angle"] - ang[b]) if np.isfinite(bm["angle"]) else np.nan
        g_err.append(ge); b_err.append(be); a_err.append(ae)
        g_ok_l.append(bool(np.isfinite(ge) and ge <= PASS_GAIN_DB))
        b_ok_l.append(bool(np.isfinite(be) and be <= max(1.0, PASS_BW_REL * wid[b])))
        a_ok_l.append(bool(np.isfinite(ae) and ae <= max(1.0, PASS_ANG_REL * wid[b])))

    gain_ok, bw_ok, ang_ok = all(g_ok_l), all(b_ok_l), all(a_ok_l)
    beam_ok = gain_ok and bw_ok and ang_ok
    sll_ok = bool(np.isfinite(f["sll"]) and f["sll"] <= sll + PASS_SLL_DB)
    null_ok = all(
        len(f["null_angles"]) > 0 and np.min(np.abs(f["null_angles"] - nang[j])) <= PASS_NULL_DEG
        for j in range(nn))
    sll_margin = float(sll + PASS_SLL_DB - f["sll"]) if np.isfinite(f["sll"]) else -10.0
    return dict(beam_ok=beam_ok, sll_ok=sll_ok, null_ok=null_ok, sll_margin=sll_margin,
                all_ok=beam_ok and sll_ok and null_ok,
                gain_ok=gain_ok, bw_ok=bw_ok, ang_ok=ang_ok,
                g_err=g_err, b_err=b_err, a_err=a_err, nb=nb, nn=nn)


def reports(Pm, Xs):
    w_ = Pm[:, :MAX_N] + 1j * Pm[:, MAX_N:2 * MAX_N]
    return [spec_report(w_[k], Xs[k]) for k in range(len(Xs))]


# ============================================================
# TF PATTERN + REFERENCE-FREE SPEC LOSS
# ============================================================
def pattern_and_peak(w_re, w_im):
    re = tf.matmul(w_re, A_re) - tf.matmul(w_im, A_im)
    im = tf.matmul(w_re, A_im) + tf.matmul(w_im, A_re)
    af = tf.sqrt(re * re + im * im + 1e-12)
    peak = tf.reduce_max(af, axis=1, keepdims=True) + 1e-12
    return af / peak, peak


def to_db(a):
    return 20.0 * tf.math.log(tf.maximum(a, DB_FLOOR)) / tf.math.log(10.0)


def decode_spec(x):
    nb = tf.round(x[:, 1:2] * MAX_B)
    ang = x[:, 3:7] * 90.0
    wid = x[:, 7:11] * 20.0
    gain = x[:, 11:15] * GAIN_SCALE
    sll = x[:, 18:19] * 20.0
    bmask = tf.cast(tf.range(MAX_B, dtype=tf.float32)[None, :] < nb, tf.float32)
    return ang, wid, gain, sll, bmask


def sample_weights(x):
    if not N_WEIGHTS:
        return None
    n = tf.round(x[:, 0] * MAX_N)
    w = tf.ones_like(n)
    for nv, wv in N_WEIGHTS.items():
        w = tf.where(tf.equal(n, float(nv)), tf.fill(tf.shape(n), float(wv)), w)
    return w


def _ratio(num, den, sw=None):
    if sw is not None:
        s = sw[:, None]
        return tf.reduce_sum(num * s) / (tf.reduce_sum(den * s) + 1e-6)
    return tf.reduce_sum(num) / (tf.reduce_sum(den) + 1e-6)


def spec_loss_terms(pred_db, x, sw=None):
    """beam gain (plateau mean, relative to strongest beam), -3 dB width (soft), sidelobe excess."""
    tau = BW_TAU_VAR
    ang, wid, gain, sll, bmask = decode_spec(x)

    d = tf.abs(theta_deg - ang[:, :, None])
    half = wid[:, :, None] / 2
    in_beam = tf.cast(d <= half, tf.float32) * bmask[:, :, None]
    near = tf.cast(d <= half + TRANS_DEG, tf.float32) * bmask[:, :, None]

    pw = tf.pow(10.0, pred_db / 10.0)
    pmean = tf.reduce_sum(pw[:, None, :] * in_beam, axis=2) / (tf.reduce_sum(in_beam, axis=2) + 1e-6)
    mean_db = 10.0 * tf.math.log(pmean + 1e-12) / tf.math.log(10.0)
    ref_db = tf.reduce_max(mean_db + 100.0 * (bmask - 1.0), axis=1, keepdims=True)
    l_beam = _ratio(tf.square(mean_db - ref_db - gain) * bmask, bmask, sw)

    dm = d + 1e3 * (1.0 - bmask[:, :, None])
    own = tf.one_hot(tf.argmin(dm, axis=1), MAX_B, axis=1)
    win = tf.cast(d <= 1.5 * half + BW_WIN_MARGIN, tf.float32) * own * bmask[:, :, None]

    level = tf.stop_gradient(mean_db) - 3.0
    soft = tf.sigmoid((pred_db[:, None, :] - level[:, :, None]) / tau)
    bw_soft = tf.reduce_sum(soft * win, axis=2) * THETA_STEP_LOSS
    tol = tf.maximum(1.0, BW_TOL_FRAC * wid)
    e_bw = tf.nn.relu(tf.abs(bw_soft - wid) - BW_MARGIN * tol) / tol
    l_bw = _ratio(tf.square(e_bw) * bmask, bmask, sw)

    side = tf.reduce_prod(1.0 - near, axis=1)
    over = tf.nn.relu(pred_db - (sll + SLL_TOL))
    l_sll = _ratio(tf.square(over) * side, side, sw)
    return l_beam, l_sll, l_bw


TERM_NAMES = ["pattern", "weight", "beam", "sll", "bw"]


def loss_terms(y_true, y_pred, x, lw):
    p_re, p_im = y_pred[:, :MAX_N], y_pred[:, MAX_N:]
    p_pat, _ = pattern_and_peak(p_re, p_im)
    p_db = to_db(p_pat)

    zero = tf.constant(0.0)
    l_pattern, l_weight = zero, zero
    if lw["db"] > 0 or lw["lin"] > 0:
        t_pat, _ = pattern_and_peak(y_true[:, :MAX_N], y_true[:, MAX_N:])
        t_db = to_db(t_pat)
        l_pattern = (lw["db"] * tf.reduce_mean(tf.square(p_db - t_db))
                     + lw["lin"] * tf.reduce_mean(tf.square(p_pat - t_pat)))
    if lw["wgt"] > 0:
        l_weight = lw["wgt"] * tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y_true), axis=1))

    l_beam, l_sll, l_bw = spec_loss_terms(p_db, x, sw=sample_weights(x))
    return dict(pattern=l_pattern, weight=l_weight, beam=lw["beam"] * l_beam,
                sll=lw["sll"] * l_sll, bw=lw["bw"] * l_bw)


# ============================================================
# LAYERS: sector geometry, null projection, features
# ============================================================
def sector_u(x):
    """Beam sector edges / centre in u = sin(theta) space."""
    deg = np.pi / 180.0
    ang = x[:, 3:7] * 90.0
    wid = x[:, 7:11] * 20.0
    ulo = tf.sin(tf.clip_by_value(ang - wid / 2.0, -90.0, 90.0) * deg)
    uhi = tf.sin(tf.clip_by_value(ang + wid / 2.0, -90.0, 90.0) * deg)
    return ulo, uhi, 0.5 * (ulo + uhi)


def layer_masks(x):
    n = tf.round(x[:, 0:1] * MAX_N)
    nb = tf.round(x[:, 1:2] * MAX_B)
    nn = tf.round(x[:, 2:3] * MAX_NUL)
    nmask = tf.cast(tf.range(MAX_N, dtype=tf.float32)[None, :] < n, tf.float32)
    bmask = tf.cast(tf.range(MAX_B, dtype=tf.float32)[None, :] < nb, tf.float32)
    jmask = tf.cast(tf.range(MAX_NUL, dtype=tf.float32)[None, :] < nn, tf.float32)
    idx = tf.range(MAX_N, dtype=tf.float32)[None, :, None]
    return n, nmask, bmask, jmask, idx


def null_project(w, x, n, nmask, jmask, idx):
    """w' = w - conj(C)(C^T conj(C) + lam I)^-1 C^T w  ->  AF(null angles) = 0."""
    nang = x[:, 15:18] * 90.0
    phi = np.pi * idx * tf.sin(nang[:, None, :] * (np.pi / 180.0))
    m = nmask[:, :, None] * jmask[:, None, :]
    C = tf.complex(tf.cos(phi) * m, tf.sin(phi) * m)
    r = tf.einsum("bn,bnj->bj", w, C)
    G = tf.matmul(C, tf.math.conj(C), transpose_a=True)
    lam = tf.cast(NULL_PROJ_REG * n[:, :, None] * tf.eye(MAX_NUL)[None], tf.complex64)
    z = tf.linalg.solve(G + lam, r[:, :, None])
    corr = tf.matmul(tf.math.conj(C), z)[:, :, 0]
    w2 = w - corr
    return tf.concat([tf.math.real(w2), tf.math.imag(w2)], axis=1)


class SpecFeatures(tf.keras.layers.Layer):
    """19 scaled specs + physics features (sin/cos of angle, sector edges in u-space, ...)."""
    def call(self, x):
        n_el = x[:, 0:1] * MAX_N
        nb = tf.round(x[:, 1:2] * MAX_B)
        bmask = tf.cast(tf.range(MAX_B, dtype=tf.float32)[None, :] < nb, tf.float32)
        rad = x[:, 3:7] * 90.0 * (np.pi / 180.0)
        wid = x[:, 7:11] * 20.0
        s = tf.sin(rad) * bmask
        c = tf.cos(rad) * bmask
        taper = wid * n_el * tf.cos(rad) / 100.0 * bmask
        inv_n = 32.0 / n_el
        ulo, uhi, uc = sector_u(x)
        du = (uhi - ulo) * n_el / 16.0
        return tf.concat([x, s, c, taper, inv_n, ulo * bmask, uhi * bmask, uc * bmask, du * bmask], axis=1)


def _per_n(n, which):
    """(B,1) array sizes -> (B,1) value of PRIOR_PARAMS[N][which]  (0 = kappa, 1 = kaiser offset)."""
    v32, v64, v128 = (tf.constant(PRIOR_PARAMS[k][which], tf.float32) for k in (32, 64, 128))
    return tf.where(n < 48.0, v32, tf.where(n < 96.0, v64, v128))


# ============================================================
# ANALYTIC PRIOR  (label generator; contains NO trainable parameters)
# ============================================================
class PriorWeights(tf.keras.layers.Layer):
    """x (B,19) -> w (B,256) = [Re | Im].
    Per beam:  sinc(width) x Kaiser(SLL) x carrier exp(-j*pi*k*u_c), scaled to the requested gain;
    beams are summed, normalised, then the exact null projection is applied.
    A deterministic function of x: one spec -> one weight vector."""

    def _sector_prior(self, x, n, nmask, bmask):
        ang = x[:, 3:7] * 90.0
        wid = x[:, 7:11] * 20.0
        gain = tf.clip_by_value(x[:, 11:15] * GAIN_SCALE, -40.0, 5.0)
        sll = x[:, 18:19] * 20.0
        ulo, uhi, uc = sector_u(x)
        kap = _per_n(n, 0)
        d = 0.5 * (uhi - ulo) + kap / n

        k = tf.range(MAX_N, dtype=tf.float32)[None, :, None]
        m = k - 0.5 * (n[:, :, None] - 1.0)
        xx = np.pi * m * d[:, None, :]
        small = tf.abs(xx) < 1e-4
        xs = tf.where(small, tf.ones_like(xx), xx)
        s_ = d[:, None, :] * tf.where(small, 1.0 - xx * xx / 6.0, tf.sin(xs) / xs)

        att = tf.clip_by_value(-sll + _per_n(n, 1), 15.0, 90.0)
        beta = tf.where(att > 50.0, 0.1102 * (att - 8.7),
                        tf.where(att > 21.0,
                                 0.5842 * tf.pow(tf.maximum(att - 21.0, 1e-6), 0.4) + 0.07886 * (att - 21.0),
                                 tf.zeros_like(att)))
        r = tf.clip_by_value(2.0 * m / tf.maximum(n[:, :, None] - 1.0, 1.0), -1.0, 1.0)
        arg = beta[:, :, None] * tf.sqrt(tf.maximum(1.0 - r * r, 0.0))
        kw = tf.math.bessel_i0(arg) / tf.math.bessel_i0(beta)[:, :, None]

        phase = -np.pi * k * uc[:, None, :]
        env = s_ * kw * nmask[:, :, None]
        wr = tf.cos(phase) * env
        wi = tf.sin(phase) * env

        wr2 = tf.reshape(tf.transpose(wr, [0, 2, 1]), [-1, MAX_N])
        wi2 = tf.reshape(tf.transpose(wi, [0, 2, 1]), [-1, MAX_N])
        pre = tf.matmul(wr2, A_re) - tf.matmul(wi2, A_im)
        pim = tf.matmul(wr2, A_im) + tf.matmul(wi2, A_re)
        pw2 = pre * pre + pim * pim
        if PRIOR_PLATEAU_NORM:
            sec = tf.cast(tf.abs(theta_deg - ang[:, :, None]) <= wid[:, :, None] / 2.0, tf.float32)
            sec = tf.reshape(sec, [-1, tf.shape(pw2)[1]])
            pk = tf.sqrt(tf.reduce_sum(pw2 * sec, axis=1) / (tf.reduce_sum(sec, axis=1) + 1e-6) + 1e-12)
        else:
            pk = tf.sqrt(tf.reduce_max(pw2, axis=1) + 1e-12)
        pk = tf.reshape(pk, [-1, MAX_B])
        scale = tf.pow(10.0, gain / 20.0) * bmask / pk

        w0r = tf.reduce_sum(wr * scale[:, None, :], axis=2)
        w0i = tf.reduce_sum(wi * scale[:, None, :], axis=2)
        return tf.complex(w0r, w0i)

    def call(self, x):
        n, nmask, bmask, jmask, idx = layer_masks(x)
        w0 = self._sector_prior(x, n, nmask, bmask)
        nrm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(w0)), axis=1, keepdims=True) + 1e-9)
        w = w0 / tf.cast(nrm, tf.complex64)
        return null_project(w, x, n, nmask, jmask, idx)


_prior_layer = None


def prior_weights(Xs, chunk=1024):
    """Analytic design for every row of Xs -> (K, 256).  Reads the CURRENT PRIOR_PARAMS."""
    global _prior_layer
    if _prior_layer is None:
        _prior_layer = PriorWeights(name="prior_only")
    outs = [_prior_layer(tf.constant(Xs[s0:s0 + chunk])).numpy() for s0 in range(0, len(Xs), chunk)]
    return np.concatenate(outs)


def unit_norm_rows(P_):
    w = P_[:, :MAX_N] + 1j * P_[:, MAX_N:]
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
    return np.concatenate([w.real, w.imag], axis=1).astype(np.float32)


def tune_prior_per_n(X, N_arr, train_rows, tune_rows=900, seed=0,
                     kappas=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7), offs=(5.0, 7.0, 9.0, 11.0)):
    """Grid search of (kappa, kaiser_off) separately for N = 32 / 64 / 128 on TRAIN rows only."""
    rows = train_rows[np.random.RandomState(seed).permutation(len(train_rows))[:tune_rows]]
    Ns_t = N_arr[rows]
    best = {n_: (-1e9, PRIOR_PARAMS[n_]) for n_ in (32, 64, 128)}
    for kp in kappas:
        for off in offs:
            for n_ in PRIOR_PARAMS:
                PRIOR_PARAMS[n_] = (kp, off)
            rr = reports(prior_weights(X[rows]), X[rows])
            parts = []
            for n_ in (32, 64, 128):
                s = [r for r, nn_ in zip(rr, Ns_t) if nn_ == n_]
                if not s:
                    continue
                all_pct = 100 * np.mean([r["all_ok"] for r in s])
                marg = float(np.mean(np.clip([r["sll_margin"] for r in s], -10, 10)))
                sc_ = all_pct + 0.5 * marg
                parts.append(f"N{n_}: all {all_pct:.0f}% marg {marg:+.1f}")
                if sc_ > best[n_][0]:
                    best[n_] = (sc_, (kp, off))
            print(f"[prior tune] kappa {kp} off {off} | " + " | ".join(parts))
    for n_ in (32, 64, 128):
        PRIOR_PARAMS[n_] = best[n_][1]
    print("[prior tune] chosen per N: " + " | ".join(
        f"N={n_}: kappa {PRIOR_PARAMS[n_][0]}, off {PRIOR_PARAMS[n_][1]}" for n_ in (32, 64, 128)))


# ============================================================
# STUDENT NETWORK (no prior inside): learned envelopes x geometric carrier
# ============================================================
class BuildEnvWeights(tf.keras.layers.Layer):
    """The net outputs one real envelope per beam (B, 4*N).  The carrier exp(-j*pi*k*u_c) of each beam
    (progressive phase = pure steering geometry) is fixed by the requested angle/width.  Envelope x carrier
    is summed over beams and the exact null projection is applied."""
    def call(self, inputs):
        raw, x = inputs
        n, nmask, bmask, jmask, idx = layer_masks(x)
        _, _, uc = sector_u(x)
        env = tf.transpose(tf.reshape(raw, [-1, MAX_B, MAX_N]), [0, 2, 1])
        env = env * nmask[:, :, None] * bmask[:, None, :] / tf.sqrt(n)[:, :, None]
        phase = -np.pi * idx * uc[:, None, :]
        wr = tf.reduce_sum(env * tf.cos(phase), axis=2)
        wi = tf.reduce_sum(env * tf.sin(phase), axis=2)
        return null_project(tf.complex(wr, wi), x, n, nmask, jmask, idx)


def build_student(blocks=6, tag="stu"):
    inp = tf.keras.Input(shape=(INPUT_DIM,), name=f"spec_{tag}")
    feat = SpecFeatures(name=f"features_{tag}")(inp)
    h = tf.keras.layers.Dense(WIDTH)(feat)
    for _ in range(blocks):
        y_ = tf.keras.layers.LayerNormalization()(h)
        y_ = tf.keras.layers.Dense(2 * WIDTH, activation="gelu")(y_)
        y_ = tf.keras.layers.Dropout(DROPOUT)(y_)
        y_ = tf.keras.layers.Dense(WIDTH)(y_)
        h = tf.keras.layers.Add()([h, y_])
    h = tf.keras.layers.LayerNormalization()(h)
    raw = tf.keras.layers.Dense(MAX_B * MAX_N, activation="linear", name=f"raw_{tag}",
                                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(h)
    out = BuildEnvWeights(name=f"weights_{tag}")([raw, inp])
    return tf.keras.Model(inp, out)


def net_predict(m, Xs, chunk=1024):
    return np.concatenate([m(tf.constant(Xs[s0:s0 + chunk]), training=False).numpy()
                           for s0 in range(0, len(Xs), chunk)])