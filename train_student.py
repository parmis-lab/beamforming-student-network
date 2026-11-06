"""
train_student.py  --  trains the STUDENT from prior_dataset_v1.npz (made once by make_prior_dataset.py).

  input : 19 specs                         (X)
  label : analytic-prior weights           (Y)      <- distillation target
  loss  : weights MSE + pattern MSE + reference-free spec loss (beam / sll / bw)
  net   : MLP -> per-beam envelopes x geometric carrier -> null projection   (no prior inside)
"""
import pickle
import random
import time

import numpy as np
import tensorflow as tf

import bf_common as bf

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "prior_dataset_v2.npz"

FILTER_BAD_LABELS = True      # drop training rows whose prior label does not pass the checker
MIN_SLL_MARGIN = None         # e.g. 0.5 -> also require >= 0.5 dB SLL margin on the label (None = off)

BATCH_SIZE, LR = 256, 1e-3
EPOCHS, ANNEAL = 40, 10       # cosine LR schedule, no early stop; tau (soft -3 dB width) anneals 3 -> 1 over ANNEAL epochs
BLOCKS = 6
SEED = 42

LW_STUDENT = dict(db=1 / 400.0, lin=1.0, wgt=1.0, beam=1 / 10.0, sll=1 / 10.0, bw=0.3)

np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# ============================================================
# LOAD
# ============================================================
d = np.load(DATA_PATH)
for n_, (kp, off) in zip((32, 64, 128), d["prior_params"]):
    bf.PRIOR_PARAMS[n_] = (float(kp), float(off))        # (only matters if the prior is evaluated again)

X_tr, Y_tr, ok_tr, marg_tr = d["X_tr"], d["Y_tr"], d["ok_tr"], d["marg_tr"]
X_va, Y_va, ok_va, marg_va = d["X_va"], d["Y_va"], d["ok_va"], d["marg_va"]
X_te, Y_te_prior, Yref_te = d["X_te"], d["Y_te"], d["Yref_te"]
N_te, nb_te = d["N_te"], d["nb_te"]


def keep_mask(ok, marg):
    m = np.ones(len(ok), bool)
    if FILTER_BAD_LABELS:
        m &= ok
    if MIN_SLL_MARGIN is not None:
        m &= marg >= MIN_SLL_MARGIN
    return m


k_tr, k_va = keep_mask(ok_tr, marg_tr), keep_mask(ok_va, marg_va)
print(f"train rows kept {int(k_tr.sum())}/{len(k_tr)} | val rows kept {int(k_va.sum())}/{len(k_va)} | test {len(X_te)} (never filtered)")
N_tr_all = np.rint(X_tr[:, 0] * bf.MAX_N).astype(int)
print("  dropped per N: " + " | ".join(f"N={v}: {int(((N_tr_all == v) & ~k_tr).sum())}" for v in (32, 64, 128)))
X_tr, Y_tr, X_va, Y_va = X_tr[k_tr], Y_tr[k_tr], X_va[k_va], Y_va[k_va]


# ============================================================
# TRAIN
# ============================================================
def batches(Xa, Ya, shuffle):
    order = np.random.permutation(len(Xa)) if shuffle else np.arange(len(Xa))
    for s0 in range(0, len(order), BATCH_SIZE):
        bi = order[s0:s0 + BATCH_SIZE]
        yield tf.constant(Xa[bi]), tf.constant(Ya[bi]), len(bi)


def train_model(model, Xt, Yt, Xv, Yv, lw, epochs, anneal, tag="student"):
    opt = tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=bf.WEIGHT_DECAY)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            yp = model(xb, training=True)
            t = bf.loss_terms(yb, yp, xb, lw)
            loss = tf.add_n(list(t.values()))
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, bf.GRAD_CLIP)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(xb, yb):
        t = bf.loss_terms(yb, model(xb, training=False), xb, lw)
        return tf.stack([tf.add_n(list(t.values()))] + [t[k] for k in bf.TERM_NAMES])

    best, best_w = np.inf, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        bf.BW_TAU_VAR.assign(1.0 + 2.0 * max(0.0, 1.0 - ep / anneal))
        lr = LR * (0.02 + 0.98 * 0.5 * (1.0 + np.cos(np.pi * (ep - 1) / epochs)))
        opt.learning_rate.assign(lr)

        tl = tn = 0.0
        for xb, yb, nb_ in batches(Xt, Yt, True):
            tl += float(train_step(xb, yb)) * nb_
            tn += nb_
        vt, vn = np.zeros(1 + len(bf.TERM_NAMES)), 0.0
        for xb, yb, nb_ in batches(Xv, Yv, False):
            vt += eval_step(xb, yb).numpy() * nb_
            vn += nb_
        vt /= vn
        va_l = float(vt[0])

        if ep >= anneal and va_l < best:               # tau still changing before `anneal` -> val not comparable
            best, best_w = va_l, model.get_weights()

        if ep == 1 or ep % 5 == 0:
            print(f"[{tag}] epoch {ep:3d} | train {tl / tn:.5f} | val {va_l:.5f} | lr {lr:.1e} | "
                  f"tau {float(bf.BW_TAU_VAR):.2f} | {time.time() - t0:.0f}s")
            print("          val terms: " + " ".join(f"{k} {v:.4f}" for k, v in zip(bf.TERM_NAMES, vt[1:])))
    if best_w is not None:
        model.set_weights(best_w)
    print(f"[{tag}] done: best val {best:.5f} | time {time.time() - t0:.0f}s")


model = bf.build_student(BLOCKS)
print(f"\n=== STUDENT (no prior inside) | params {model.count_params():,} | train rows {len(X_tr)}")
train_model(model, X_tr, Y_tr, X_va, Y_va, LW_STUDENT, EPOCHS, ANNEAL)
model.save_weights("beamforming_student_v2.weights.h5")

# ============================================================
# EVALUATION (exact MATLAB-port checker)
# ============================================================
print("\n=== TEST EVALUATION")
P_st = bf.net_predict(model, X_te)
cand = {"reference": Yref_te, "prior only": Y_te_prior, "student": P_st}
rep = {k: bf.reports(v, X_te) for k, v in cand.items()}


def pc(a):
    a = np.asarray(a, dtype=float)
    return 100.0 * float(a.mean()) if a.size else float("nan")


print("\nSPEC SATISFACTION (test)")
for name, r in rep.items():
    f = lambda key: pc([x[key] for x in r])
    print(f"  {name:12s} beam_ok {f('beam_ok'):5.1f}% | bw_ok {f('bw_ok'):5.1f}% | sll_ok {f('sll_ok'):5.1f}% | "
          f"null_ok {f('null_ok'):5.1f}% | ALL {f('all_ok'):5.1f}%")

print("\nBY N  (ALL)")
for name, r in rep.items():
    print(f"  {name:12s} " + " | ".join(
        f"N={v}: {pc([r[k]['all_ok'] for k in np.where(N_te == v)[0]]):.1f}%" for v in np.unique(N_te)))

print("\nBY NUMBER OF BEAMS  (ALL)")
for name, r in rep.items():
    print(f"  {name:12s} " + " | ".join(
        f"{v} beams: {pc([r[k]['all_ok'] for k in np.where(nb_te == v)[0]]):.1f}%" for v in np.unique(nb_te)))

print("\nFAILURE CAUSES BY N (student)")
for v in np.unique(N_te):
    s = np.where(N_te == v)[0]
    g = lambda key: pc([rep["student"][k][key] for k in s])
    print(f"  N={v:3d}: gain {g('gain_ok'):5.1f}% | bw {g('bw_ok'):5.1f}% | angle {g('ang_ok'):5.1f}% | "
          f"sll {g('sll_ok'):5.1f}% | null {g('null_ok'):5.1f}% | ALL {g('all_ok'):5.1f}%")

# SLL failures that are really "checker found no SLL group at all" (sll = NaN)
f_st = [r for r in rep["student"] if not r["sll_ok"]]
print(f"\n[student] SLL failures: {len(f_st)} (sll_margin == -10 means the checker found NO sidelobe group: "
      f"{sum(1 for r in f_st if r['sll_margin'] <= -10 + 1e-6)})")

# weights similarity (informative only: many weight vectors give the same pattern)
wt = Yref_te[:, :bf.MAX_N] + 1j * Yref_te[:, bf.MAX_N:]
wp = P_st[:, :bf.MAX_N] + 1j * P_st[:, bf.MAX_N:]
wq = Y_te_prior[:, :bf.MAX_N] + 1j * Y_te_prior[:, bf.MAX_N:]
print("\n[student] mean |dB err| vs prior-label pattern "
      f"{np.abs(bf.db_np(wp) - bf.db_np(wq)).mean():.2f} dB | "
      f"w_sim vs prior label {np.mean([bf.wsim(wq[k, :N_te[k]], wp[k, :N_te[k]]) for k in range(len(X_te))]):.3f}")
print("[student] mean |dB err| vs MATLAB reference pattern "
      f"{np.abs(bf.db_np(wp) - bf.db_np(wt)).mean():.2f} dB | "
      f"w_sim vs reference {np.mean([bf.wsim(wt[k, :N_te[k]], wp[k, :N_te[k]]) for k in range(len(X_te))]):.3f}")

with open("test_predictions_student_v2.pkl", "wb") as f:
    pickle.dump(dict(X=X_te, P_student=P_st, P_prior=Y_te_prior, Y_ref=Yref_te, N=N_te, nb=nb_te,
                     output_format="[Re(w)(128) | Im(w)(128)]"), f)
print("\nsaved: beamforming_student_v2.weights.h5, test_predictions_student_v2.pkl")