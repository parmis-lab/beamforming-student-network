"""
demo_student.py -- feed ONE unseen test spec to the trained student and show everything.

Usage (put it next to bf_common.py, prior_dataset_v2.npz and beamforming_student_v2.weights.h5):
    python demo_student.py            # a random test spec
    python demo_student.py pass       # a random test spec that the student passes
    python demo_student.py fail       # a random test spec that the student fails
    python demo_student.py 123        # test row 123 (reproducible)

It prints the requested spec, the measured result of the exact checker, the student's weights,
the inference time, and it saves demo_output.png (pattern + weights) and demo_weights.csv.
"""
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import tensorflow as tf

import bf_common as bf

DATA_PATH = "prior_dataset_v2.npz"
WEIGHTS_PATH = "beamforming_student_v2.weights.h5"
BLOCKS = 6                       # same as BLOCKS in train_student.py
SHOW = True                      # open the plot window (the PNG is always saved)
OUT_PNG = "demo_output.png"
OUT_CSV = "demo_weights.csv"
LATENCY_REPEATS = 100


def cplx(P):
    return P[:, :bf.MAX_N] + 1j * P[:, bf.MAX_N:2 * bf.MAX_N]


def pick_row(arg, model, X, rng):
    if arg.isdigit():
        k = int(arg)
        if not 0 <= k < len(X):
            raise SystemExit(f"row must be between 0 and {len(X) - 1}")
        return k
    if arg in ("pass", "fail"):
        P_all = bf.net_predict(model, X)
        ok = np.array([r["all_ok"] for r in bf.reports(P_all, X)])
        pool = np.where(ok if arg == "pass" else ~ok)[0]
        return int(rng.choice(pool))
    return int(rng.integers(len(X)))


def measure_latency(model, x1):
    """Returns (weights, first_call_seconds, steady_state_seconds) for ONE spec."""
    fn = tf.function(lambda x: model(x, training=False))
    xt = tf.constant(x1)
    t0 = time.perf_counter()
    w = fn(xt).numpy()                      # first call includes graph tracing
    first = time.perf_counter() - t0
    for _ in range(10):
        fn(xt).numpy()
    t0 = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        w = fn(xt).numpy()
    return w, first, (time.perf_counter() - t0) / LATENCY_REPEATS


def main(arg="random"):
    rng = np.random.default_rng()
    d = np.load(DATA_PATH)
    X, P_pr, P_ref, N_te = d["X_te"], d["Y_te"], d["Yref_te"], d["N_te"]

    model = bf.build_student(BLOCKS)
    model.load_weights(WEIGHTS_PATH)

    k = pick_row(arg, model, X, rng)
    x = X[k]
    N = int(N_te[k])
    nb = int(round(x[1] * bf.MAX_B)); nn = int(round(x[2] * bf.MAX_NUL))
    ang, wid, gain = x[3:7][:nb] * 90, x[7:11][:nb] * 20, x[11:15][:nb] * bf.GAIN_SCALE
    nang, sll = x[15:18][:nn] * 90, x[18] * 20

    print("=" * 78)
    print(f"TEST ROW {k}  (this spec was never used for training)")
    print(f"  array: N = {N} elements | {nb} beam(s) | {nn} null(s) | requested SLL = {sll:.1f} dB")
    for b in range(nb):
        print(f"  beam {b + 1}: angle {ang[b]:+6.1f} deg | width {wid[b]:5.1f} deg | relative gain {gain[b]:+5.1f} dB")
    for j in range(nn):
        print(f"  null {j + 1}: angle {nang[j]:+6.1f} deg")

    # ---- the network: ONE spec in, 256 numbers out, with timing ----
    out, first, lat = measure_latency(model, x[None])
    w = cplx(out)[0]
    print("-" * 78)
    print("STUDENT OUTPUT")
    print(f"  input: 19 numbers -> output: 256 numbers = Re(w) and Im(w) of {N} complex weights")
    print(f"  inference time, one spec: {1e3 * lat:.2f} ms  (average of {LATENCY_REPEATS} calls; "
          f"the very first call took {1e3 * first:.0f} ms because TensorFlow builds the graph)")
    print("  first weights (amplitude, phase):")
    for n in range(min(6, N)):
        print(f"    w[{n}] = {abs(w[n]):.4f} at {np.degrees(np.angle(w[n])):+7.1f} deg")

    # ---- the exact checker ----
    t0 = time.perf_counter()
    rep = bf.spec_report(w, x)
    chk_ms = 1e3 * (time.perf_counter() - t0)
    db = bf.db_raw(w)
    f = bf.extract_features_matlab(db, bf.theta_np, ang, wid)
    gabs = np.array([b["gain"] for b in f["beams"]], float)
    gmax = np.nanmax(gabs) if np.isfinite(gabs).any() else np.nan

    print("-" * 78)
    print(f"EXACT CHECKER (port of the MATLAB extraction, {chk_ms:.2f} ms)")
    for b in range(nb):
        bm = f["beams"][b]
        g_ok = bool(np.isfinite(rep["g_err"][b]) and rep["g_err"][b] <= bf.PASS_GAIN_DB)
        w_ok = bool(np.isfinite(rep["b_err"][b]) and rep["b_err"][b] <= max(1.0, bf.PASS_BW_REL * wid[b]))
        a_ok = bool(np.isfinite(rep["a_err"][b]) and rep["a_err"][b] <= max(1.0, bf.PASS_ANG_REL * wid[b]))
        print(f"  beam {b + 1}: angle {bm['angle']:+6.1f} (req {ang[b]:+6.1f}) | width {bm['bw']:5.2f} (req {wid[b]:5.1f}) | "
              f"gain {bm['gain'] - gmax:+5.2f} (req {gain[b]:+5.1f})  ->  "
              f"{'OK' if (g_ok and w_ok and a_ok) else 'FAIL'}")
    for j in range(nn):
        dist = float(np.min(np.abs(f["null_angles"] - nang[j]))) if len(f["null_angles"]) else float("nan")
        print(f"  null {j + 1}: nearest real null {dist:5.2f} deg away (limit {bf.PASS_NULL_DEG:.0f} deg)")
    if np.isfinite(f["sll"]):
        print(f"  SLL: measured {f['sll']:.1f} dB, requested {sll:.1f} dB, limit {sll + bf.PASS_SLL_DB:.1f} dB "
              f"(margin {rep['sll_margin']:+.1f} dB)")
    else:
        print(f"  SLL: the checker found NO sidelobe group (3 peaks within 2 dB) -> counted as a failure")
    bad = [name for name, key in (("gain", "gain_ok"), ("bandwidth", "bw_ok"), ("angle", "ang_ok"),
                                  ("SLL", "sll_ok"), ("null", "null_ok")) if not rep[key]]
    verdict = "PASS (all criteria met)" if rep["all_ok"] else "FAIL on: " + ", ".join(bad)
    print(f"  VERDICT: {verdict}")

    wt, wr = cplx(P_pr)[k], cplx(P_ref)[k]
    print("-" * 78)
    print("COMPARISON")
    print(f"  weight similarity with the analytic teacher : {bf.wsim(w[:N], wt[:N]):.3f}")
    print(f"  weight similarity with the MATLAB reference : {bf.wsim(w[:N], wr[:N]):.3f}  "
          f"(low is normal: many different weights give the same pattern)")
    print("=" * 78)

    # ---- files ----
    np.savetxt(OUT_CSV, np.column_stack([np.arange(N), w[:N].real, w[:N].imag, np.abs(w[:N]), np.degrees(np.angle(w[:N]))]),
               delimiter=",", header="n,re,im,amplitude,phase_deg", comments="")

    # ---- figure ----
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])

    ax.plot(bf.theta_np, bf.db_raw(wr), color="tab:green", lw=0.8, alpha=0.7)
    ax.plot(bf.theta_np, bf.db_raw(wt), color="0.45", lw=1.0, ls="--")
    ax.plot(bf.theta_np, db, color="tab:blue", lw=1.4)
    for b in range(nb):
        lo, hi = ang[b] - wid[b] / 2, ang[b] + wid[b] / 2
        ax.axvspan(lo, hi, color="orange", alpha=0.18, lw=0)
        if np.isfinite(gmax):
            ax.hlines(gmax + gain[b], lo, hi, colors="red", lw=2.5)
            bm = f["beams"][b]
            if np.isfinite(bm["L"]) and np.isfinite(bm["R"]):
                ax.plot([bm["L"], bm["R"]], [bm["gain"] - 3.0] * 2, "v", color="tab:red", ms=6)
    ax.axhline(sll, color="purple", ls="--", lw=1)
    ax.axhline(sll + bf.PASS_SLL_DB, color="purple", ls=":", lw=1)
    for a in nang:
        ax.axvline(a, color="k", ls=":", lw=1)
    if np.isfinite(f["sll"]):
        ax.plot(f["sll_angle"], f["sll"], "x", color="purple", ms=9, mew=2)
    ax.set_xlim(-90, 90); ax.set_ylim(-70, 5); ax.grid(alpha=0.25)
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("dB (normalised to peak)")
    ax.set_title(f"Test row {k}: N={N}, {nb} beam(s), {nn} null(s), SLL spec {sll:.0f} dB -> {verdict}   "
                 f"[student inference {1e3 * lat:.2f} ms]",
                 fontsize=10, color=("tab:green" if rep["all_ok"] else "tab:red"))
    ax.legend(handles=[
        Line2D([], [], color="tab:blue", lw=1.4, label="student"),
        Line2D([], [], color="0.45", lw=1.0, ls="--", label="analytic teacher"),
        Line2D([], [], color="tab:green", lw=0.8, label="MATLAB reference"),
        Patch(color="orange", alpha=0.25, label="requested beam sector"),
        Line2D([], [], color="red", lw=2.5, label="requested beam level"),
        Line2D([], [], color="tab:red", marker="v", ls="", label="measured -3 dB edges"),
        Line2D([], [], color="purple", ls="--", label="requested SLL (dotted: +2 dB limit)"),
        Line2D([], [], color="k", ls=":", label="requested null")], loc="lower left", fontsize=7, ncol=2)

    n_idx = np.arange(N)
    a1.plot(n_idx, np.abs(wt[:N]), color="0.45", ls="--", lw=1, label="teacher")
    a1.plot(n_idx, np.abs(w[:N]), color="tab:blue", lw=1.3, label="student")
    a1.set_xlabel("element index n"); a1.set_ylabel("|w_n|"); a1.grid(alpha=0.25); a1.legend(fontsize=8)
    a1.set_title("weight amplitudes", fontsize=9)
    a2.plot(n_idx, np.degrees(np.angle(wt[:N])), color="0.45", ls="--", lw=1)
    a2.plot(n_idx, np.degrees(np.angle(w[:N])), color="tab:blue", lw=1.3)
    a2.set_xlabel("element index n"); a2.set_ylabel("phase (deg)"); a2.grid(alpha=0.25)
    a2.set_title("weight phases", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"saved {OUT_PNG} and {OUT_CSV}")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "random")