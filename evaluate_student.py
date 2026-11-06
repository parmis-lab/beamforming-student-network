"""
evaluate_student.py  --  run AFTER train_student.py

Answers three questions about the trained student, using ONLY test specs (never seen in training):
  1. Did it really generalise?   (train-vs-test gap, kNN "memorise" baseline, teacher comparison, CIs)
  2. What do its patterns look like?   (PNG plots with the spec drawn on top of every pattern)
  3. How long does one unseen spec take?   (network, prior formula, checker)

Outputs: console report + PNG files in ./eval_plots
"""
import os
import platform
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist
import tensorflow as tf

import bf_common as bf

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "prior_dataset_v2.npz"
WEIGHTS_PATH = "beamforming_student_v2.weights.h5"
BLOCKS = 6                       # must be the same as BLOCKS in train_student.py
OUT_DIR = "eval_plots_v2"
N_TRAIN_CHECK = 1000             # real training rows used for the train-vs-test comparison
N_EXAMPLES_PER_N = 2             # random PASS examples per N in the "pass" figure
N_FAIL_EXAMPLES = 6
N_FOUR_BEAM = 4
TIMING_REPEATS = 200
SEED = 0

THETA = bf.theta_np
CRIT = (("gain", "gain_ok"), ("bw", "bw_ok"), ("angle", "ang_ok"), ("sll", "sll_ok"), ("null", "null_ok"))


# ============================================================
# SMALL HELPERS
# ============================================================
def cplx(P):
    return P[:, :bf.MAX_N] + 1j * P[:, bf.MAX_N:2 * bf.MAX_N]


def wilson(k, n, z=1.96):
    """95 % Wilson confidence interval of a pass rate, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def pc(a):
    a = np.asarray(a, dtype=float)
    return 100.0 * float(a.mean()) if a.size else float("nan")


def all_ok(rep):
    return np.array([r["all_ok"] for r in rep], dtype=bool)


def knn_baseline(X_tr, Y_tr, X_te):
    """'Memorising' baseline: for every test spec copy the weights of the NEAREST training spec
    (same N and same number of beams, distance on angles / widths / gains / nulls / SLL)."""
    N_tr = np.rint(X_tr[:, 0] * bf.MAX_N).astype(int)
    nb_tr = np.rint(X_tr[:, 1] * bf.MAX_B).astype(int)
    N_te_ = np.rint(X_te[:, 0] * bf.MAX_N).astype(int)
    nb_te_ = np.rint(X_te[:, 1] * bf.MAX_B).astype(int)
    P = np.zeros((len(X_te), 2 * bf.MAX_N), np.float32)
    for n in np.unique(N_te_):
        for b in np.unique(nb_te_[N_te_ == n]):
            ti = np.where((N_te_ == n) & (nb_te_ == b))[0]
            ri = np.where((N_tr == n) & (nb_tr == b))[0]
            if len(ri) == 0:
                ri = np.where(N_tr == n)[0]
            D = cdist(X_te[ti, 3:], X_tr[ri, 3:])
            P[ti] = Y_tr[ri[D.argmin(axis=1)]]
    return P


# ============================================================
# PLOTS
# ============================================================
def draw_panel(ax, x, w_st, w_pr, w_ref, rep, show_legend=False):
    """One test spec: student / prior / MATLAB-reference pattern + the requested spec drawn on top."""
    N = int(round(x[0] * bf.MAX_N)); nb = int(round(x[1] * bf.MAX_B)); nn = int(round(x[2] * bf.MAX_NUL))
    ang = x[3:7][:nb] * 90; wid = x[7:11][:nb] * 20; gain = x[11:15][:nb] * bf.GAIN_SCALE
    nang = x[15:18][:nn] * 90; sll = x[18] * 20

    db_ref = bf.db_raw(w_ref); db_pr = bf.db_raw(w_pr); db_st = bf.db_raw(w_st)
    ax.plot(THETA, db_ref, color="tab:green", lw=0.8, alpha=0.7)
    ax.plot(THETA, db_pr, color="0.45", lw=1.0, ls="--")
    ax.plot(THETA, db_st, color="tab:blue", lw=1.3)

    f = bf.extract_features_matlab(db_st, THETA, ang, wid)
    gabs = np.array([b["gain"] for b in f["beams"]], float)
    gmax = np.nanmax(gabs) if np.isfinite(gabs).any() else np.nan
    for b in range(nb):
        lo, hi = ang[b] - wid[b] / 2, ang[b] + wid[b] / 2
        ax.axvspan(lo, hi, color="orange", alpha=0.18, lw=0)                 # requested sector
        if np.isfinite(gmax):
            ax.hlines(gmax + gain[b], lo, hi, colors="red", lw=2.5)          # requested plateau level
            bm = f["beams"][b]
            if np.isfinite(bm["L"]) and np.isfinite(bm["R"]):                # measured -3 dB crossings
                ax.plot([bm["L"], bm["R"]], [bm["gain"] - 3.0] * 2, "v", color="tab:red", ms=5)
    ax.axhline(sll, color="purple", ls="--", lw=1)                           # requested SLL
    ax.axhline(sll + bf.PASS_SLL_DB, color="purple", ls=":", lw=1)           # pass limit
    for a in nang:
        ax.axvline(a, color="k", ls=":", lw=1)                               # requested nulls
    if np.isfinite(f["sll"]):
        ax.plot(f["sll_angle"], f["sll"], "x", color="purple", ms=8, mew=2)  # measured SLL

    ax.set_xlim(-90, 90); ax.set_ylim(-70, 5)
    ax.grid(alpha=0.25)
    bad = [k for k, key in CRIT if not rep[key]]
    status = "PASS" if rep["all_ok"] else "FAIL: " + ", ".join(bad)
    errs = []
    for key, unit in (("g_err", "dB"), ("b_err", "deg")):
        v = np.array(rep[key], float)
        if v.size and np.isfinite(v).any():
            errs.append(f"max {key[0]}-err {np.nanmax(v):.2f}{unit}")
    nosll = rep["sll_margin"] <= -10 + 1e-6
    errs.append("SLL: no group found" if nosll else f"SLL margin {rep['sll_margin']:+.1f} dB")
    ax.set_title(f"N={N}, {nb} beam(s), {nn} null(s), SLL spec {sll:.0f} dB  ->  {status}\n" + " | ".join(errs),
                 fontsize=9, color=("tab:green" if rep["all_ok"] else "tab:red"))
    ax.set_ylabel("dB (normalised to peak)", fontsize=8)
    if show_legend:
        ax.legend(handles=[
            Line2D([], [], color="tab:blue", lw=1.3, label="student"),
            Line2D([], [], color="0.45", lw=1.0, ls="--", label="prior (teacher)"),
            Line2D([], [], color="tab:green", lw=0.8, label="MATLAB reference"),
            Patch(color="orange", alpha=0.25, label="requested beam sector"),
            Line2D([], [], color="red", lw=2.5, label="requested beam level"),
            Line2D([], [], color="tab:red", marker="v", ls="", label="measured -3 dB edges"),
            Line2D([], [], color="purple", ls="--", label="requested SLL (dotted: +2 dB limit)"),
            Line2D([], [], color="k", ls=":", label="requested null")],
            loc="lower left", fontsize=6.5, ncol=2)


def make_figure(idx_list, title, fname, X, P_st, P_pr, P_ref, rep_st):
    if len(idx_list) == 0:
        return
    W_st, W_pr, W_ref = cplx(P_st), cplx(P_pr), cplx(P_ref)
    n = len(idx_list); rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 3.7 * rows), squeeze=False)
    for j, k in enumerate(idx_list):
        draw_panel(axes[j // 2, j % 2], X[k], W_st[k], W_pr[k], W_ref[k], rep_st[k], show_legend=(j == 0))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("theta (deg)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def make_error_figure(rep_st, rep_pr, X, fname):
    """Per-beam errors divided by their pass tolerance (<= 1 passes) + SLL margin histogram."""
    def collect(rep):
        g, b, a = [], [], []
        for r, x in zip(rep, X):
            wid = x[7:11] * 20
            for j in range(r["nb"]):
                g.append(r["g_err"][j] / bf.PASS_GAIN_DB)
                b.append(r["b_err"][j] / max(1.0, bf.PASS_BW_REL * wid[j]))
                a.append(r["a_err"][j] / max(1.0, bf.PASS_ANG_REL * wid[j]))
        return [np.asarray(v, float) for v in (g, b, a)]

    cs, cp = collect(rep_st), collect(rep_pr)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    bins = np.linspace(0, 2, 41)
    for ax, name, vs, vp in zip(axes.ravel()[:3], ("gain error / 1 dB", "bandwidth error / tolerance", "angle error / tolerance"), cs, cp):
        for v, lab, col in ((vp, "prior (teacher)", "0.45"), (vs, "student", "tab:blue")):
            fin = v[np.isfinite(v)]
            ax.hist(np.clip(fin, 0, 2), bins=bins, histtype="step", lw=1.6, color=col,
                    label=f"{lab}: {int((~(v <= 1)).sum())} beams > tol")
        ax.axvline(1.0, color="red", ls="--")
        ax.set_title(name + "   (left of the red line = pass)"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    bins2 = np.linspace(-10.5, 20, 62)
    for rep, lab, col in ((rep_pr, "prior (teacher)", "0.45"), (rep_st, "student", "tab:blue")):
        m = np.array([r["sll_margin"] for r in rep])
        ax.hist(np.clip(m, -10.5, 20), bins=bins2, histtype="step", lw=1.6, color=col,
                label=f"{lab}: {int((m <= -10 + 1e-6).sum())} with no SLL group (-10)")
    ax.axvline(0, color="red", ls="--")
    ax.set_title("SLL margin (dB) = spec + 2 dB - measured;  < 0 fails, -10 = checker found no group")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def make_bar_figure(oks, N_te, nb_te, fname):
    colors = {"prior (teacher)": "0.55", "student": "tab:blue", "kNN (copy nearest train spec)": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, arr, lab in ((axes[0], N_te, "N"), (axes[1], nb_te, "number of beams")):
        cats = np.unique(arr)
        wdt = 0.8 / len(oks)
        for i, (name, ok) in enumerate(oks.items()):
            vals = [pc(ok[arr == c]) for c in cats]
            ax.bar(np.arange(len(cats)) + i * wdt, vals, wdt, label=name, color=colors.get(name))
        ax.set_xticks(np.arange(len(cats)) + wdt * (len(oks) - 1) / 2)
        ax.set_xticklabels([f"{lab}={c}  (n={int((arr == c).sum())})" for c in cats], fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("ALL pass rate (%)"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def pick_examples(rep_st, N_te, nb_te, rng):
    ok = all_ok(rep_st)
    passes = []
    for n in (32, 64, 128):
        c = np.where((N_te == n) & ok)[0]
        if len(c):
            passes += list(rng.choice(c, min(N_EXAMPLES_PER_N, len(c)), replace=False))
    f = np.where(~ok)[0]
    fails = list(rng.choice(f, min(N_FAIL_EXAMPLES, len(f)), replace=False)) if len(f) else []
    q = np.where(nb_te == 4)[0]
    fours = list(rng.choice(q, min(N_FOUR_BEAM, len(q)), replace=False)) if len(q) else []
    return passes, fails, fours


# ============================================================
# TIMING
# ============================================================
def timing_report(model, X_te, P_st):
    print(f"\n[timing] python {platform.python_version()} | TF {tf.__version__} | "
          f"devices: {[d.device_type for d in tf.config.list_physical_devices()]} | {platform.processor()[:60]}")
    fn = tf.function(lambda x: model(x, training=False))

    def run(x):
        return fn(tf.constant(x)).numpy()

    for i in range(10):
        run(X_te[i:i + 1])                                      # warm-up (graph tracing)
    t0 = time.perf_counter()
    for i in range(TIMING_REPEATS):
        j = i % len(X_te)
        run(X_te[j:j + 1])
    lat = (time.perf_counter() - t0) / TIMING_REPEATS
    print(f"  student, ONE spec at a time (latency)     : {1e3 * lat:8.2f} ms / spec")
    for bs in (64, 256, len(X_te)):
        xb = X_te[:bs]
        run(xb); run(xb)
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            run(xb)
        per = (time.perf_counter() - t0) / reps / len(xb)
        print(f"  student, batch {len(xb):5d} (throughput)         : {1e3 * per:8.3f} ms / spec")

    bf.prior_weights(X_te[:1])
    t0 = time.perf_counter()
    for i in range(50):
        bf.prior_weights(X_te[i:i + 1])
    pri = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    bf.prior_weights(X_te)
    pri_b = (time.perf_counter() - t0) / len(X_te)
    print(f"  analytic prior formula, one spec          : {1e3 * pri:8.2f} ms / spec  (batch: {1e3 * pri_b:.3f} ms / spec)")

    W = cplx(P_st)
    n_chk = min(200, len(X_te))
    t0 = time.perf_counter()
    for k in range(n_chk):
        bf.spec_report(W[k], X_te[k])
    chk = (time.perf_counter() - t0) / n_chk
    print(f"  exact checker (extraction + pass/fail)    : {1e3 * chk:8.2f} ms / spec")
    print(f"  student + checker, one spec               : {1e3 * (lat + chk):8.2f} ms / spec")
    print("  (compare with the time of ONE MATLAB optimisation (WLS/CLS/RWLS/CVX) for the same spec -- time it with tic/toc)")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = np.load(DATA_PATH)
    X_te, P_pr, P_ref = d["X_te"], d["Y_te"], d["Yref_te"]
    N_te, nb_te = d["N_te"], d["nb_te"]
    X_tr, Y_tr, src_tr, ok_tr = d["X_tr"], d["Y_tr"], d["src_tr"], d["ok_tr"]

    print("=== 0. ARE THE TEST SPECS REALLY UNSEEN?")
    ov = np.intersect1d(d["train_spec_ids"], d["test_spec_ids"])
    print(f"  spec ids in train AND test: {len(ov)}  (must be 0) | train specs {len(d['train_spec_ids'])} | "
          f"val {len(d['val_spec_ids'])} | test {len(d['test_spec_ids'])}")
    print("  test rows per N: " + " | ".join(f"N={v}: {int((N_te == v).sum())}" for v in np.unique(N_te)))
    print("  test rows per #beams: " + " | ".join(f"{v}: {int((nb_te == v).sum())}" for v in np.unique(nb_te)))

    model = bf.build_student(BLOCKS)
    model.load_weights(WEIGHTS_PATH)
    P_st = bf.net_predict(model, X_te)
    rep_st = bf.reports(P_st, X_te)
    rep_pr = bf.reports(P_pr, X_te)
    ok_st, ok_pr = all_ok(rep_st), all_ok(rep_pr)

    print("\n=== 1. HEADLINE (test, exact MATLAB-port checker)")
    for name, ok in (("prior (teacher)", ok_pr), ("student", ok_st),
                     ("student + checker, prior as fallback", ok_st | ok_pr)):
        lo, hi = wilson(int(ok.sum()), len(ok))
        print(f"  {name:38s} ALL {100 * ok.mean():5.1f}%   95% CI [{lo:.1f}, {hi:.1f}]   (n={len(ok)})")

    print("\n=== 2. OVERFITTING CHECK: same net on training specs vs test specs")
    real = np.where(src_tr == 0)[0]
    sel = rng.choice(real, min(N_TRAIN_CHECK, len(real)), replace=False)
    ok_trn = all_ok(bf.reports(bf.net_predict(model, X_tr[sel]), X_tr[sel]))
    print(f"  student on {len(sel)} TRAIN specs: {100 * ok_trn.mean():5.1f}%   (prior on the same rows: {100 * ok_tr[sel].mean():.1f}%)")
    print(f"  student on {len(ok_st)} TEST  specs: {100 * ok_st.mean():5.1f}%   (prior on the same rows: {100 * ok_pr.mean():.1f}%)")
    print("  -> a small gap means it generalises; a big train>>test gap would mean memorisation.")

    print("\n=== 3. MEMORISATION BASELINE: copy the weights of the nearest training spec (kNN)")
    P_knn = knn_baseline(X_tr, Y_tr, X_te)
    rep_knn = bf.reports(P_knn, X_te)
    ok_knn = all_ok(rep_knn)
    print(f"  kNN ALL {100 * ok_knn.mean():5.1f}%   vs student {100 * ok_st.mean():5.1f}%   vs prior {100 * ok_pr.mean():5.1f}%")
    print("  -> if the student is far above kNN, it learned the spec->weights mapping instead of recalling samples.")

    print("\n=== 4. WHO FAILS?  (student vs its teacher, per N)")
    for n in (32, 64, 128):
        m = N_te == n
        print(f"  N={n:3d}: both ok {int((ok_st & ok_pr & m).sum()):4d} | teacher ok, student FAIL {int((~ok_st & ok_pr & m).sum()):3d} | "
              f"student ok, teacher FAIL {int((ok_st & ~ok_pr & m).sum()):3d} | both fail {int((~ok_st & ~ok_pr & m).sum()):3d}")
    print("  failure causes of the student per N:")
    for n in (32, 64, 128):
        s = np.where(N_te == n)[0]
        g = lambda key: pc([rep_st[k][key] for k in s])
        print(f"    N={n:3d}: gain {g('gain_ok'):5.1f}% | bw {g('bw_ok'):5.1f}% | angle {g('ang_ok'):5.1f}% | "
              f"sll {g('sll_ok'):5.1f}% | null {g('null_ok'):5.1f}% | ALL {g('all_ok'):5.1f}%")
    nogrp = sum(1 for r in rep_st if (not r["sll_ok"]) and r["sll_margin"] <= -10 + 1e-6)
    print(f"  SLL failures with NO sidelobe group found by the checker: {nogrp} of {sum(1 for r in rep_st if not r['sll_ok'])}")

    print("\n=== 5. PLOTS")
    passes, fails, fours = pick_examples(rep_st, N_te, nb_te, rng)
    make_figure(passes, "Student on unseen TEST specs -- random PASS examples", "patterns_pass.png",
                X_te, P_st, P_pr, P_ref, rep_st)
    make_figure(fails, "Student on unseen TEST specs -- random FAIL examples", "patterns_fail.png",
                X_te, P_st, P_pr, P_ref, rep_st)
    make_figure(fours, "Student on unseen TEST specs -- random 4-beam examples", "patterns_4beams.png",
                X_te, P_st, P_pr, P_ref, rep_st)
    make_error_figure(rep_st, rep_pr, X_te, "error_histograms.png")
    make_bar_figure({"prior (teacher)": ok_pr, "student": ok_st, "kNN (copy nearest train spec)": ok_knn},
                    N_te, nb_te, "pass_rate_bars.png")

    timing_report(model, X_te, P_st)


if __name__ == "__main__":
    main()