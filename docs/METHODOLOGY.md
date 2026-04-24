# Methodology

Phrase Nudging is an inference-time control method that SLERPs the model's
next-token distribution toward phrase-level targets mined from recorded
reasoning trajectories. This document pairs the mathematical formalism
with the implementation code from `src/`.

---

## 1. Output-distribution representation

For each generated token position *t* of a trajectory we capture the
full-vocabulary softmax $p_t \in \Delta^{V-1}$ and store it as a
**coverage-based sparse top-K**: we keep exactly the smallest set of
vocab indices whose cumulative mass is $\geq \tau$ (default
$\tau = 0.999$), rather than a fixed $K$.

Math. For a distribution $p \in \Delta^{V-1}$ define
$$
K(p;\tau) \;=\; \min\left\{\, k \;:\; \sum_{i=1}^k p_{(i)} \geq \tau \,\right\}
$$
where $p_{(1)} \geq p_{(2)} \geq \ldots$ is the sorted probability.
We retain the indices of the top $K(p;\tau)$ entries.

Code (`src/sparse_utils.py`):

```python
def coverage_topk(P: np.ndarray, coverage: float = 0.999,
                   probe_m: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    V = P.shape[0]
    M = min(V, probe_m)
    idx_part = np.argpartition(-P, M - 1)[:M]
    order_local = np.argsort(-P[idx_part])
    sorted_idx = idx_part[order_local]
    sorted_p   = P[sorted_idx]
    cum = np.cumsum(sorted_p)
    total_mass = float(P.sum())
    target = coverage * total_mass
    if float(cum[-1]) >= target:
        k = int(np.searchsorted(cum, target, side="left")) + 1
        k = max(1, min(k, M))
        return sorted_idx[:k].astype(np.int32), sorted_p[:k].astype(np.float32)
    # fallback: full argsort if the probe_m window didn't reach τ
    order = np.argsort(-P)
    ...
```

A two-stage `argpartition`-then-full-sort avoids the $O(V \log V)$ cost
of argsorting a $V{\sim}152\text{k}$ vector at every decode step.

Per-trajectory storage is CSR-like: an `indptr` array of cumulative
counts, plus flat `idx` and `val` buffers. To reconstruct the dense
unit-norm amplitude vector at any position (needed at decode time for
SLERP targets), we slice and re-normalize:

```python
def reconstruct_amp(indptr, idx, val, pos, V, buf):
    buf.fill(0.0)
    s, e = int(indptr[pos]), int(indptr[pos + 1])
    amp_slice = np.sqrt(np.maximum(val[s:e], 0.0))
    n = float(np.sqrt((amp_slice * amp_slice).sum()))
    if n > 0: amp_slice = amp_slice / n
    buf[idx[s:e]] = amp_slice.astype(np.float32, copy=False)
    return buf
```

The $\sqrt{\cdot}$ turns probability mass into amplitude; renormalization
compensates for the $(1-\tau)$ missing tail.

---

## 2. Amplitude sphere and Bhattacharyya coefficient

Any probability distribution $p \in \Delta^{V-1}$ maps to an amplitude
vector $a = \sqrt{p}$ on the unit sphere $S^{V-1}$, since
$\sum_i p_i = 1 \iff \sum_i a_i^2 = 1$. On this manifold the
**Bhattacharyya coefficient** between $p$ and $q$ is exactly the
inner product of their amplitudes,
$$
\mathrm{BC}(p, q) \;=\; \sum_i \sqrt{p_i q_i} \;=\; \langle a, b \rangle \;\in\; [0, 1].
$$

This is a cosine on $S^{V-1}$, so the angle
$\theta = \arccos\mathrm{BC}(p,q) \in [0, \pi/2]$ is the geodesic
distance on the amplitude hypersphere — a proper Riemannian metric.

---

## 3. Phrase-window distance

We compare W-token *phrases* rather than single tokens. Given two
trajectories $A, B$ with amplitude sequences
$a^A_0, \ldots$ and $a^B_0, \ldots$, an offset $k$ and a phrase
center $t$, the W-phrase Bhattacharyya is
$$
\mathrm{BC}_W(A, B;\, t, k) \;=\; \frac{1}{W}\sum_{s=0}^{W-1}
\langle a^A_{t+s},\, a^B_{t+k+s} \rangle.
$$

This is the **per-token average** of aligned BCs across the window —
not a cosine of window-mean amps, and not a BC of mean distributions.
The per-token formulation preserves temporal alignment (aligned
positions contribute independently) and is the only form that permits
the matmul + rolling-mean fast path below.

The stored distance is the raw angle
$$
d(t, k) \;=\; \arccos\mathrm{BC}_W(A, B;\, t, k) \;\in\; [0, \pi/2]
$$
(in radians). We use $\theta$ rather than the sin-angle
$\sqrt{1 - \mathrm{BC}_W^2}$ or the Hellinger distance
$\sqrt{1 - \mathrm{BC}_W}$ because both saturate near $\theta = \pi/2$
— derivatives vanish — which would cause the downstream Gaussian
kernel to collapse all near-orthogonal pairs. Raw $\theta$ is linear
in the angle and keeps the tail discriminative.

---

## 4. Fast pairwise implementation

Let $\Phi_X \in \mathbb{R}^{T_X \times V}$ be the stacked per-token
amplitude matrix of trajectory $X$ (unit-norm rows). Then
$$
G \;=\; \Phi_A \Phi_B^\top, \qquad G[a, b] \;=\; \langle a^A_a, a^B_b \rangle,
$$
is the dense per-token BC matrix. For each offset $k$, the values on
the $k$-diagonal of $G$ are the BCs we need to average, and a
length-$W$ rolling mean via cumulative-sum gives
$\mathrm{BC}_W(t, k)$ at every valid $t$.

Code (`src/allpairs_bc.py`, `pair_dist_fast`):

```python
# Per-token BC matrix: G[a, b] = <a_A[a], a_B[b]> = BC(p_A[a], p_B[b]).
G = (Phi_A @ Phi_B.T).astype(np.float32).toarray()
G = np.clip(G, 0.0, 1.0)

d = np.full((len(k_range), T_A), np.nan, dtype=np.float32)
for ki, k in enumerate(k_range):
    i_lo = max(0, -k); i_hi = min(T_A, T_B - k)
    if i_hi - i_lo < W: continue

    # k-diagonal: diag_k[j] = G[i_lo+j, i_lo+j+k]
    rows = np.arange(i_lo, i_hi); cols = rows + k
    diag_k = G[rows, cols]

    # Length-W rolling mean via cumsum.
    cs = np.concatenate([[0.0], np.cumsum(diag_k, dtype=np.float64)])
    rolling_sum = cs[W:] - cs[:-W]
    bc_per_window = np.clip((rolling_sum / W).astype(np.float32), 0.0, 1.0)

    # Raw angular distance θ = arccos(BC_W), radians in [0, π/2].
    d[ki, i_lo:i_lo + bc_per_window.shape[0]] = \
        np.arccos(bc_per_window).astype(np.float32)
```

One sparse matmul per pair + $O(|k\text{-range}| \cdot T)$ for the
diagonal extraction and rolling sums. For $T\sim 600$ and
$|k\text{-range}|\sim 200$, a pair runs well under a second.

---

## 5. Affinity kernel and spectral clustering

We enumerate phrase windows $(traj, center)$ at stride $S$ across each
trajectory and assemble the $N \times N$ inter-phrase angular-distance
matrix $\Theta_{ij}$ from the stored per-pair $d(t, k)$. We then form
a Gaussian affinity
$$
A_{ij} \;=\; \exp\!\left(-\frac{\Theta_{ij}^2}{2\sigma^2}\right)
$$
with $\sigma$ in radians (same units as $\Theta$). The affinity width
is a manifold-scale knob: $\sigma \ll \mathrm{med}(\Theta)$ gives a
near-disconnected graph, $\sigma \gg \mathrm{med}(\Theta)$ oversmooths,
and the intermediate regime exposes cluster structure.

**Spectral clustering.** Given $A$, form the degree matrix
$D = \mathrm{diag}(A\,\mathbf{1})$ and the normalized Laplacian
$$
L \;=\; I - D^{-1/2} A D^{-1/2}.
$$

The bottom-$k$ eigenvectors of $L$ embed each phrase in
$\mathbb{R}^k$; k-means on that embedding produces cluster labels.

Code (`src/spectral_cluster.py`):

```python
A = np.exp(-THETA ** 2 / (2 * args.sigma ** 2)).astype(np.float64)
np.fill_diagonal(A, 1.0)

sc = SpectralClustering(n_clusters=args.k, affinity="precomputed",
                         random_state=args.seed, assign_labels="kmeans")
labels = sc.fit_predict(A)
```

A $(\sigma, k)$ sweep ranked by silhouette + temporal coherence + size
balance picks the most characteristic clustering for a given trajectory
corpus.

---

## 6. Phrase library

A cluster $c$ is operationalized as a **library of phrase amp-sequences**:
for every member phrase $(traj, center)$ of cluster $c$, the W-token
window of amplitude vectors around $center$ is stored in the same
coverage-based CSR layout as the raw lens. Step 4 materializes this
per-cluster library from the clustering output.

---

## 7. SLERP nudging at decode time

At decode step $t$, let the model's natural next-token distribution be
$p_t$, with amplitude $a_t = \sqrt{p_t} \in S^{V-1}$. Pick a target
amplitude $a^*$ from the current cluster's phrase library (a sampled
phrase's amplitude at the appropriate within-phrase position). The
angle between them is
$$
\theta \;=\; \arccos\langle a_t, a^* \rangle,
$$
and for a small $\alpha \in [0, 1]$ the **spherical linear interpolation**
$$
a'_t \;=\; \frac{\sin((1-\alpha)\theta)}{\sin\theta}\, a_t
        \;+\; \frac{\sin(\alpha\theta)}{\sin\theta}\, a^*
$$
stays on the unit sphere. The element-wise square $p'_t = (a'_t)^2$
is a valid probability distribution — we sample the next token from it
via inverse-CDF.

Code (`src/run_controller.py`):

```python
def slerp(a, a_target, alpha):
    dot = float(np.clip(np.dot(a, a_target), -1.0, 1.0))
    theta = float(np.arccos(dot))
    if theta < 1e-6:
        return a.astype(np.float32)
    sin_t = np.sin(theta)
    c1 = np.sin((1 - alpha) * theta) / sin_t
    c2 = np.sin(alpha * theta) / sin_t
    return (c1 * a + c2 * a_target).astype(np.float32)
```

and the full nudge step is:

```python
P = softmax(logits, temp=nudge_temp)         # optional pre-SLERP temperature
a = np.sqrt(P).astype(np.float32)
a_new = slerp(a, target, alpha)
P_new = (a_new * a_new).astype(np.float64)
P_new /= P_new.sum()
cdf = np.cumsum(P_new); cdf[-1] = 1.0
tok = int(np.searchsorted(cdf, rng.random(), side="right"))
```

The `--alpha` and `--nudge-temp` knobs give separate control over *how
far* we move on the sphere ($\alpha$, measured as a fraction of the
total angle $\theta$) and *how sharp* the starting amplitude is
(temperature scaling of logits before the $\sqrt{\cdot}$).

---

## 8. Control modes

Two ways to assemble the target-cluster sequence:

### (a) Span-plan schedule (steps 5 / 6 / 7)

A plan is a list of spans $(\mathrm{kind}, c, L, \alpha)$:
- *kind* ∈ {nudge_sched, nudge_commit, free}
- *c* is a cluster id (None for free)
- *L* is the span length in tokens
- $\alpha$ is a per-span SLERP alpha (None → fall back to global `--alpha`)

The grammar is a comma-separated block list:
`"c4:50+35, c1:50+35@0.02, c11:50+35, c7:50+35@0.05"`. Each block
contributes a nudge span of length $n$ followed by a free span of length
$f$; `@α` is optional.

Code (`src/run_controller.py`, `gen_controlled`):

```python
for span in plan:
    kind, cid, length, span_alpha = span
    cur_phrase = None  # fresh phrase at every span boundary
    for s_in_span in range(length):
        logits = get_logits()
        if kind in ("nudge_sched", "nudge_commit", "nudge"):
            if cur_phrase is None:
                cur_phrase = phrase_seqs_by_cid[cid][rng.randrange(...)]
            pos = int(s_in_span * W / max(1, length))
            target = reconstruct_amp(...)
            P = softmax(logits, temp=nudge_temp)
            a = np.sqrt(P)
            al = span_alpha if span_alpha is not None else (
                 commit_alpha if kind == "nudge_commit" else alpha)
            a_new = slerp(a, target, al)
            ... sample from a_new ** 2 ...
        else:  # free
            tok = sample_from_logits(logits, free_temp, rng, V)
```

### (b) Markov process (step 8)

Let $\mathcal{C} = [c_0, \ldots, c_{N-1}]$ be an ordered list of clusters
plus a "free" state indexed $N$. A transition matrix
$T \in \mathbb{R}^{(N+1)\times(N+1)}$ with non-negative rows summing to 1
describes a Markov chain over these states. At every decode step we
sample the next state
$$
s_{t+1} \;\sim\; T[s_t, \cdot]
$$
and then act: if $s_{t+1} < N$ we SLERP toward $c_{s_{t+1}}$ at a fixed
$\alpha$; if $s_{t+1} = N$ we free-sample. Phrase targets are resampled
on state entry and every $W$ consecutive steps within the same state.

Code (`examples/step8_markov_nudge.py`, `generate_markov`):

```python
for _ in range(n_tokens):
    row = T[state]
    cdf_states = np.cumsum(row); cdf_states[-1] = 1.0
    new_state = int(np.searchsorted(cdf_states, rng.random(), side="right"))
    if new_state > N: new_state = N
    if new_state != state:
        cur_phrase = None; steps_in_state = 0
    state = new_state

    if state < N:
        if cur_phrase is None or (steps_in_state > 0 and steps_in_state % W == 0):
            cur_phrase = phrase_seqs_by_cid[clusters[state]][rng.randrange(...)]
        pos = steps_in_state % W
        target = reconstruct_amp(...)
        P = softmax(logits, temp=nudge_temp)
        ... SLERP + sample ...
    else:
        tok = sample_from_logits(logits, free_temp, rng, V_llm)
    steps_in_state += 1
```

Any span-plan is representable as a Markov chain with block-tridiagonal
transitions; the converse is not true (Markov chains can mix
stochastically and have absorbing states).

---

## 9. Force-commit normalization

On reasoning benchmarks requiring a `\boxed{...}` answer, a decode
failure can come from two distinct causes: (1) the model never derived
the answer, or (2) it derived the answer but got stuck in a hedge loop
and never emitted `\boxed{}`. To score (1) vs (2) separately, we
optionally inject a forcing prefix after the natural decode stops:

```python
def force_box_emission(llm, prefix, max_answer_tokens, rng, temp):
    prefix_tokens = list(llm.tokenize(prefix.encode(), add_bos=False))
    llm.eval(prefix_tokens)
    appended = list(prefix_tokens)
    for _ in range(max_answer_tokens):
        logits = get_logits()
        tok = sample_from_logits(logits, temp, rng, V)
        if tok == eos: break
        appended.append(tok); llm.eval([tok])
        piece = llm.detokenize([tok]).decode(...)
        if "}" in piece: break   # the closing brace of \boxed{...}
    return appended
```

The cutoff position (where natural decode stops and forcing begins) is
controlled by `--force-at`:
- `end`: after full `--n-tokens` / plan
- `post-schedule`: right before any commit cycle
- `<int>` or `<pct>%`: absolute or relative position

Records in the output JSONL include a `forced: bool` field so the
natural-vs-forced rate can be analyzed post-hoc.
