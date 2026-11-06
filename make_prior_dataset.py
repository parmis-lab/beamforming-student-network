"""
make_prior_dataset.py  --  run ONCE.

  MATLAB .mat (extracted specs)  ->  X (19 features)
  X  ->  analytic prior  ->  Y_prior (labels)  ->  checker  ->  ok / sll_margin flags
  everything is saved to prior_dataset_v1.npz, so train_student.py never has to rebuild the prior.

Optional: SYNTH_COUNTS adds synthetic training specs drawn with the SAME rules as the MATLAB
generateRandomSpec (only those the prior itself can satisfy are kept).  Val / test stay 100 % real.
"""
import time

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

import bf_common as bf

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = (
    r"C:\Users\USER\Desktop\Beamforming_Amplitude_Phase_Estimation"
    r"\MATLAB\beamforming_training_data.mat"
)
OUT_PATH = "prior_dataset_v2.npz"          # v1 stays untouched as a fallback

SEED = 42
TEST_SIZE, VAL_SIZE = 0.15, 0.15
VAL_MAX, EVAL_MAX = 4000, 3000          # same sub-sampling as the v6 script -> same val / test rows

TUNE_PRIOR = False                       # False: use the values stored in bf.PRIOR_PARAMS (found by v6 tuning)
TUNE_ROWS = 900

SYNTH_COUNTS = {32: 30000, 64: 15000, 128: 15000}   # synthetic TRAINING specs per N (before the checker filter); {} = off

np.random.seed(SEED)


# ============================================================
# LOAD THE MATLAB DATASET  (inputs = EXTRACTED specs)
# ============================================================
def vec(x):
    return np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()


def load_mat(path):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    ds = np.asarray(mat["datasetStruct"], dtype=object).reshape(-1)
    X_list, Y_list, ids, Ns, nbs = [], [], [], [], []
    skipped = 0
    for i, r in enumerate(ds):
        N = int(r.N)
        nb = int(r.numBeams_extracted)
        nn = int(r.numNulls_extracted)
        if not (1 <= nb <= bf.MAX_B) or not (0 <= nn <= bf.MAX_NUL) or not (1 <= N <= bf.MAX_N):
            raise ValueError(f"Row {i}: invalid N/numBeams/numNulls = {N}/{nb}/{nn}")

        x = bf.build_x(N, nb, nn, r.beamAngle_extracted, r.beamWidth_extracted,
                       r.beamGainRel_extracted, r.nullAngle_extracted, float(r.SLL_extracted))

        amp, ph = vec(r.w_amp), vec(r.w_phase)
        if len(amp) != N or len(ph) != N:
            raise ValueError(f"Row {i}: weight length mismatch (N={N})")
        w = amp * np.exp(1j * ph)
        y = np.zeros(2 * bf.MAX_N, np.float32)
        y[:N] = w.real
        y[bf.MAX_N:bf.MAX_N + N] = w.imag

        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            skipped += 1
            continue
        X_list.append(x); Y_list.append(y)
        ids.append(int(r.SpecIdx)); Ns.append(N); nbs.append(nb)
    print(f"rows used: {len(X_list)} (skipped: {skipped})")
    return np.stack(X_list), np.stack(Y_list), np.asarray(ids), np.asarray(Ns), np.asarray(nbs)


X, Y_ref, spec_ids, N_arr, nb_arr = load_mat(DATASET_PATH)
print("[N counts] " + " | ".join(f"N={v}: {int((N_arr == v).sum())}" for v in np.unique(N_arr)))

# ---- split by spec (identical to v6) ----
unique_specs = np.unique(spec_ids)
tr_ids, tmp_ids = train_test_split(unique_specs, test_size=TEST_SIZE + VAL_SIZE, random_state=SEED)
va_ids, te_ids = train_test_split(tmp_ids, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), random_state=SEED)
tr = np.where(np.isin(spec_ids, tr_ids))[0]
va = np.where(np.isin(spec_ids, va_ids))[0]
te = np.where(np.isin(spec_ids, te_ids))[0]
if VAL_MAX is not None and len(va) > VAL_MAX:
    va = np.sort(np.random.RandomState(3).permutation(va)[:VAL_MAX])
if EVAL_MAX is not None and len(te) > EVAL_MAX:
    te = np.sort(np.random.RandomState(1).permutation(te)[:EVAL_MAX])
print(f"train/val/test rows: {len(tr)}/{len(va)}/{len(te)}")

# ---- sanity: checker on the MATLAB reference weights (should be ~100 %) ----
_s = tr[np.random.RandomState(2).permutation(len(tr))[:150]]
_rr = [bf.spec_report(Y_ref[i][:bf.MAX_N] + 1j * Y_ref[i][bf.MAX_N:], X[i]) for i in _s]
print(f"[sanity] checker on reference weights, 150 train rows: ALL {100 * np.mean([r['all_ok'] for r in _rr]):.1f}%")

# ============================================================
# PRIOR PARAMETERS
# ============================================================
if TUNE_PRIOR:
    bf.tune_prior_per_n(X, N_arr, tr, tune_rows=TUNE_ROWS)
print("[prior params] " + " | ".join(f"N={n}: kappa {bf.PRIOR_PARAMS[n][0]}, off {bf.PRIOR_PARAMS[n][1]}"
                                     for n in (32, 64, 128)))


# ============================================================
# LABELS = analytic design for each spec, + checker verdict on the label itself
# ============================================================
def make_labels(Xs, tag):
    t0 = time.time()
    Yp = bf.unit_norm_rows(bf.prior_weights(Xs))
    rep = bf.reports(Yp, Xs)
    ok = np.array([r["all_ok"] for r in rep], dtype=bool)
    marg = np.array([r["sll_margin"] for r in rep], dtype=np.float32)
    Ns = np.rint(Xs[:, 0] * bf.MAX_N).astype(int)
    per_n = " | ".join(f"N={v}: {100 * ok[Ns == v].mean():.1f}%" for v in (32, 64, 128) if (Ns == v).any())
    print(f"[labels {tag}] {len(Xs)} rows, label ALL-pass {100 * ok.mean():.1f}% ({per_n}) | {time.time() - t0:.0f}s")
    return Yp, ok, marg


# ---- train: real rows + mirrored rows (theta -> -theta) ----
X_real = X[tr]
X_mir = bf.mirror_x(X_real)
X_tr = np.concatenate([X_real, X_mir])
src_tr = np.concatenate([np.zeros(len(X_real), np.int8), np.ones(len(X_mir), np.int8)])   # 0 real, 1 mirror, 2 synth


# ============================================================
# OPTIONAL: synthetic specs, same rules as MATLAB generateRandomSpec
# ============================================================
def random_spec_x(N, rng):
    while True:                                   # restart if the random placement gets stuck
        nb = int(rng.integers(1, 5))              # randi([1 4])
        ang, wid, good = [], [], True
        for _ in range(nb):
            for _t in range(200):
                a = -70 + 140 * rng.random()      # [-70, 70]
                w = 6 + 16 * rng.random()         # [6, 22]
                if all(abs(a - ak) >= w / 2 + wk / 2 + 5 for ak, wk in zip(ang, wid)):
                    ang.append(a); wid.append(w)
                    break
            else:
                good = False
                break
        if not good:
            continue
        nn = int(rng.integers(1, 4))              # randi([1 3])
        nul = []
        for _ in range(nn):
            for _t in range(200):
                a = -80 + 160 * rng.random()
                if (all(abs(a - ak) >= wk / 2 + 5 for ak, wk in zip(ang, wid))
                        and all(abs(a - u) > 5 for u in nul)):
                    nul.append(a)
                    break
            else:
                good = False
                break
        if good:
            break
    gains = 10 + 10 * rng.random(nb)              # absolute gain 10..20 dB
    sll = -(13 + 27 * rng.random())               # -13 .. -40 dB
    order = np.argsort(ang)                       # real data is assumed ascending (mirror_x relies on it)
    ang = np.asarray(ang)[order]; wid = np.asarray(wid)[order]; gains = gains[order]
    nul = np.sort(np.asarray(nul))
    return bf.build_x(N, nb, nn, ang, wid, gains - gains.max(), nul, sll)   # gain -> relative to strongest


def feature_samples(Xs):
    nb = np.rint(Xs[:, 1] * bf.MAX_B).astype(int)
    nn = np.rint(Xs[:, 2] * bf.MAX_NUL).astype(int)
    bm = np.arange(bf.MAX_B)[None, :] < nb[:, None]
    nm = np.arange(bf.MAX_NUL)[None, :] < nn[:, None]
    return {"numBeams": nb.astype(float), "numNulls": nn.astype(float),
            "beamAngle": (Xs[:, 3:7] * 90)[bm], "beamWidth": (Xs[:, 7:11] * 20)[bm],
            "beamGainRel": (Xs[:, 11:15] * bf.GAIN_SCALE)[bm],
            "nullAngle": (Xs[:, 15:18] * 90)[nm], "SLL": Xs[:, 18] * 20}


def compare_distributions(X_real_, X_syn_):
    fr, fs = feature_samples(X_real_), feature_samples(X_syn_)
    print("[distribution check]  p5 / p50 / p95  (real | synthetic)")
    for k in fr:
        q = lambda a: " / ".join(f"{v:7.2f}" for v in np.percentile(a, [5, 50, 95]))
        print(f"   {k:12s} real {q(fr[k])}  |  synth {q(fs[k])}")


if sum(SYNTH_COUNTS.values()) > 0:
    rng = np.random.default_rng(SEED)
    X_syn = np.stack([random_spec_x(N, rng) for N, cnt in SYNTH_COUNTS.items() for _ in range(cnt)])
    compare_distributions(X_real, X_syn)
    Y_syn, ok_syn, marg_syn = make_labels(X_syn, "synthetic")
    keep = ok_syn
    print(f"[synthetic] kept {int(keep.sum())}/{len(keep)} (prior passes its own checker)")
    compare_distributions(X_real, X_syn[keep])
    X_tr = np.concatenate([X_tr, X_syn[keep]])
    src_tr = np.concatenate([src_tr, np.full(int(keep.sum()), 2, np.int8)])

Y_tr, ok_tr, marg_tr = make_labels(X_tr, "train")
Y_va, ok_va, marg_va = make_labels(X[va], "val")
Y_te, ok_te, marg_te = make_labels(X[te], "test (prior only)")

np.savez_compressed(
    OUT_PATH,
    X_tr=X_tr, Y_tr=Y_tr, ok_tr=ok_tr, marg_tr=marg_tr, src_tr=src_tr,
    X_va=X[va], Y_va=Y_va, ok_va=ok_va, marg_va=marg_va,
    X_te=X[te], Y_te=Y_te, ok_te=ok_te, Yref_te=Y_ref[te], N_te=N_arr[te], nb_te=nb_arr[te],
    prior_params=np.array([bf.PRIOR_PARAMS[n] for n in (32, 64, 128)], dtype=np.float32),
    train_spec_ids=tr_ids, val_spec_ids=va_ids, test_spec_ids=te_ids,
)
print(f"\nsaved: {OUT_PATH}  (train {len(X_tr)} | val {len(va)} | test {len(te)})")