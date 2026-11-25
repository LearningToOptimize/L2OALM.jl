```@meta
CurrentModule = L2OALM
```

# Theory

This page states the method as optimization problems, gives every formula the
code implements with the equation numbers of the paper, and states the
properties of the method with the hypotheses they need. Equation numbers `(n)` refer to

> S. Park and P. Van Hentenryck. *Self-Supervised Primal-Dual Learning for
> Constrained Optimization.* AAAI 2023. [arXiv:2208.09046](https://arxiv.org/abs/2208.09046).

Equations labelled `(B·)` are the row-wise box form used by the code; each is
shown to reduce to the paper's equation.

## Notation

| paper | here | meaning |
|:--|:--|:--|
| ``\mathbf x \in \mathbb R^p`` | ``p`` | instance parameters (proxy input) |
| ``\mathbf y`` | ``x`` | decision variables (primal proxy output) |
| ``f_{\mathbf x}(\mathbf y)`` | ``f(x; p)`` | objective |
| ``g_{\mathbf x}(\mathbf y) \le 0``, ``h_{\mathbf x}(\mathbf y) = 0`` | ``l \le c(x; p) \le u`` | constraints |
| ``\boldsymbol\mu \ge 0``, ``\boldsymbol\lambda`` | ``\lambda`` (one per row, signed) | multipliers (dual proxy output) |
| ``\rho``, ``\alpha``, ``\tau``, ``\rho_{\max}`` | `ρ`, `α`, `τ`, `ρmax` | penalty and its schedule |

The batch is the last array dimension: `P` is `num_parameters × B`, `X` is
`num_decision_variables × B`, `Λ` is `num_rows × B`.

## The parametric problem

The paper considers, for each instance (eq. (1)),

```math
\min_{\mathbf y \in \mathbf Y} f_{\mathbf x}(\mathbf y)
\quad\text{s.t.}\quad g_{\mathbf x, j}(\mathbf y) \le 0 \;(j \in \mathcal G),\quad
h_{\mathbf x, j}(\mathbf y) = 0 \;(j \in \mathcal H),
```

with smooth, possibly nonconvex ``f, g, h``. L2OALM writes every constraint,
variable bounds included, as a **box row**

```math
\min_x \; s\, f(x; p) \quad\text{s.t.}\quad l_i \le c_i(x; p) \le u_i, \quad i = 1, \dots, m,
\tag{B1}
```

where ``l_i, u_i \in \mathbb R \cup \{\pm\infty\}`` and ``s = +1`` for
minimisation, ``s = -1`` for maximisation (so the method always minimises
``s f``). An inequality ``g \le 0`` is a row with ``c = g``, ``u = 0``,
``l = -\infty``; an equality ``h = b`` is a row with ``l = u = b``. For a JuMP
problem the rows are `L2OCore.dual_layout(problem)` (see
[`ALMFormulation`](@ref)); for an ExaModel they are the constraints followed by
the variables.

**The variable domain ``\mathbf Y``.** In the paper ``\mathbf Y`` is "a closed
subset of ``\mathbb R^n`` defined by some bound constraints" (§3.2). It is
never penalized or dualized: eq. (5) has terms only for ``g`` and ``h``, the dual
network outputs "``\mu, \lambda`` for equalities and inequalities" (§4), and in
the AC-OPF experiments the bounds are enforced by the output layer, a convex
combination of the bounds with a Hardsigmoid (App. C.2). L2OALM follows this
when the bound map enforces the bounds ([`enforces_bounds`](@ref):
[`SigmoidBoundMap`](@ref), [`HardSigmoidBoundMap`](@ref)): the variable-bound
rows then get zero multipliers in [`project_multipliers`](@ref) and contribute
nothing to the losses (they are satisfied). With [`NoBoundMap`](@ref) the bounds
are ordinary rows with multipliers and penalties — the paper's QP benchmarks
have no variable bounds, so both choices agree there.

**Multiplier convention.** Row ``i`` carries one multiplier ``\lambda_i`` in the
normal cone of ``[l_i, u_i]`` at ``c_i``: ``\lambda_i \ge 0`` acts on the upper
bound, ``\lambda_i \le 0`` on the lower bound. The KKT conditions of (B1) are

```math
s \nabla f(x) + \nabla c(x)^\top \lambda = 0, \qquad
\lambda \in N_{[l,u]}\big(c(x)\big),
\tag{B-KKT}
```

and ``\lambda \in N_{[l,u]}(c)`` is equivalent, for any ``\rho > 0``, to

```math
c = \Pi_{[l,u]}\big(c + \lambda/\rho\big).
\tag{B-NC}
```

For a row ``g \le 0`` the paper's ``\mu`` is ``\lambda``; for a row
``c \ge l`` it is ``-\lambda``. `JuMP.dual` returns ``-\lambda`` for both
minimisation and maximisation ([`to_jump_duals`](@ref)); this is checked
against Ipopt in the tests.

## The losses

### Primal loss — eq. (5)

With the dual proxy frozen, the primal proxy minimises (eq. (5))

```math
\mathcal L_p(\mathbf y \mid \boldsymbol\mu, \boldsymbol\lambda)
= f(\mathbf y) + \boldsymbol\mu^\top g(\mathbf y) + \boldsymbol\lambda^\top h(\mathbf y)
+ \frac\rho2 \Big( \sum_{j\in\mathcal G} \max(g_j, 0)^2 + \sum_{j\in\mathcal H} h_j^2 \Big),
```

averaged over the batch. Row-wise ([`PDLAugmentation`](@ref)):

```math
\mathcal L_p = \frac1B \sum_{k=1}^B \Big( s f(x_k) + \sum_i
\big(\lambda_{ik}^+ (c_{ik} - u_i) + \lambda_{ik}^- (l_i - c_{ik})\big)
+ \frac\rho2 \sum_i v_{ik}^2 \Big),
\qquad v_{ik} = \max(l_i - c_{ik}, 0) + \max(c_{ik} - u_i, 0).
\tag{B2}
```

For ``g = c - u`` the bracket is ``\mu g`` and ``v = \max(g, 0)``; for an
equality it is ``\lambda h`` and ``v = |h|`` — exactly eq. (5). The paper writes eq. (5) per instance and notes that the
losses "generalize naturally to mini-batches" (footnote 1); L2OALM averages
over the batch. With ``\lambda = 0`` it is the penalty loss of eq. (4) with all
per-constraint coefficients ``\rho_{g,j} = \rho_{h,j} = \rho/2``, which is the
paper's remark that the zero-initialised dual network makes (5) identical to
(4).

The alternative [`PHRAugmentation`](@ref) is the classical
Powell–Hestenes–Rockafellar augmented Lagrangian

```math
\mathcal L_p^{\mathrm{PHR}} = \frac1B \sum_k \Big( s f(x_k) + \sum_i
\frac\rho2 \operatorname{dist}\big(c_{ik} + \lambda_{ik}/\rho, [l_i, u_i]\big)^2
- \frac{\lambda_{ik}^2}{2\rho} \Big).
\tag{B3}
```

Its gradient with respect to ``c`` is ``\lambda + \rho\,\sigma`` with
``\sigma`` from (B4): the primal step sees the multiplier one ALM update ahead.
The two losses coincide on equality rows
and on violated inequality rows (``c \ge u``). On satisfied inequalities they
differ: (B2) keeps the linear term ``\lambda (c - u)``, which keeps pushing the
row inside, whereas the gradient of (B3) vanishes once
``c \le u - \lambda/\rho``.

### Dual target — eq. (6)

The ALM multiplier update of eq. (6),
``\boldsymbol\mu_{k+1} = \max(\boldsymbol\mu_k + \rho\, g(\mathbf y), 0)``,
``\boldsymbol\lambda_{k+1} = \boldsymbol\lambda_k + \rho\, h(\mathbf y)``, reads
row-wise ([`dual_target`](@ref))

```math
\lambda^{k+1} = \Pi_M\big(\lambda^k + \rho\, \sigma(x, \lambda^k)\big), \qquad
\sigma(x, \lambda) = c(x) - \Pi_{[l,u]}\big(c(x) + \lambda/\rho\big).
\tag{B4}
```

For ``g = c - u``: ``\sigma = \max(g, -\mu/\rho)`` and
``\lambda + \rho\sigma = \max(\mu + \rho g, 0)``; for an equality ``\sigma = h``.
``\Pi_M`` ([`project_multipliers`](@ref)) clips
to the sign pattern of the row and to ``[-M, M]`` with `M = max_dual`.

### Dual loss — eq. (7)

The dual proxy is regressed onto the targets with (eq. (7))

```math
\mathcal L_d = \|\boldsymbol\mu - \boldsymbol\mu_{k+1}\| + \|\boldsymbol\lambda - \boldsymbol\lambda_{k+1}\|,
```

implemented as ([`dual_loss`](@ref))

```math
\mathcal L_d = \frac1B \sum_k \Big( \|(\lambda_k - t_k)_{\mathcal G}\|_\varepsilon
+ \|(\lambda_k - t_k)_{\mathcal H}\|_\varepsilon \Big), \qquad
\|v\|_\varepsilon = \sqrt{\|v\|_2^2 + \varepsilon^2} - \varepsilon,
\tag{B5}
```

where ``\mathcal H`` are the equality rows and ``\mathcal G`` all others;
with ``\varepsilon = 0``, (B5) is eq. (7). The norms are not squared, as in eq. (7); the paper leaves the
norm unspecified ("``\|\cdot\|`` represents an arbitrary norm", §3.1) and
L2OALM uses the Euclidean norm.

### Violation measure — eq. (8) — and penalty rule — eq. (9)

```math
v_k = \max_{\text{training set}} \max\big(\|h\|_\infty, \|\boldsymbol\sigma\|_\infty\big),
\qquad \sigma_j = \max(g_j, -\mu_j/\rho),
```

which is ``\max_{i,k} |\sigma_{ik}|`` with ``\sigma`` of (B4)
([`violation_measure`](@ref)). The paper
prints ``\sigma_j = \max(g_j, -\lambda_{k,j}/\rho)`` for ``j \in \mathcal G``,
i.e. with the letter of the equality multipliers; since ``j`` ranges over the
inequalities we read it as ``\mu_{k,j}``, the multiplier of ``g_j`` (as in
Andreani et al., 2008, from which the rule is taken). Then
(eq. (9), [`update_penalty`](@ref))

```math
\rho \leftarrow \min(\alpha \rho, \rho_{\max}) \quad\text{if}\quad v_k > \tau\, v_{k-1}.
\tag{B7}
```

### Algorithm 1

[`train!`](@ref) runs, for ``k = 1, \dots, K``: (i) the primal phase on (B2)
or (B3) with the dual proxy frozen; (ii) ``v_k`` by (8); (iii) a frozen copy
``\lambda_k`` of the dual proxy; (iv) the dual phase on (B5) towards (B4);
(v) the penalty rule (B7). This is the order of Algorithm 1, lines 2–10:
``v_k`` uses the dual network *before* dual learning, the dual targets use
the current ``\rho``, and ``\rho`` changes only after dual learning. The rule applies for ``k > 1`` (§4.3; ``v_0`` is not
defined), implemented as ``v_0 = \infty``. The dual proxy's last layer should
start at zero ("readily achievable by imposing the initial weights and bias
parameters of the final layer of ``D_\phi`` to be zero", §4.1), so that the
first primal phase is the penalty method.

For reference, the paper's settings (App. C): ``K = 10`` outer iterations;
500 inner epochs per phase (QP), 5000 inner iterations (QCQP), 250 inner epochs
(AC-OPF); mini-batch 200; Adam with learning rate ``10^{-4}``, decayed by 0.99
when the validation loss exceeds the best one; grids
``\tau \in \{0.5,\dots,0.9\}``, ``\rho \in \{0.1, 0.5, 1, 10\}``,
``\rho_{\max} \in \{10^3, 5\cdot 10^3, 10^4, 5\cdot 10^4\}``,
``\alpha \in \{1, 1.5, 2, 5, 10\}``.

## Properties of the method

The following facts follow directly from the formulas above.

  - **KKT points are fixed points.** If ``(x^\star, \lambda^\star)`` satisfies
    (B-KKT) for an instance ``p`` and ``\lvert\lambda^\star\rvert \le M``,
    then for every ``\rho > 0`` the dual target (B4) returns ``\lambda^\star``
    and ``v = 0``: by (B-NC), complementarity gives
    ``c^\star = \Pi(c^\star + \lambda^\star/\rho)``, so ``\sigma = 0`` and
    (B4) is the identity. If ``f`` and ``c`` are differentiable, also
    ``\nabla_x \mathcal L_p(x^\star \mid \lambda^\star) = 0`` for (B2) and
    (B3): for (B2), at a feasible ``x^\star`` the penalty gradient vanishes
    and the multiplier term has gradient ``\nabla c^\top \lambda^\star``, so
    the gradient is the KKT residual; for (B3) the gradient is
    ``s\nabla f + \nabla c^\top(\lambda^\star + \rho\sigma) =
    s\nabla f + \nabla c^\top \lambda^\star``.
  - **Penalty schedule.** With ``\alpha > 1``, ``\rho`` never decreases and
    never exceeds ``\max(\rho_0, \rho_{\max})``.
  - **Multiplier signs.** Dual targets and completed multipliers have the sign
    of their row's dual cone and satisfy ``\lvert\lambda\rvert \le M``,
    because ``\Pi_M`` projects onto that set (and is idempotent).
  - **Objective sense.** The losses of ``\max f`` and ``\min -f`` coincide.
  - **Variable bounds.** With `bounds = SigmoidBoundMap()` or
    `HardSigmoidBoundMap()` the predicted ``x`` satisfies its variable bounds
    exactly, by construction, and the variable-bound rows carry zero
    multipliers and zero dual targets, as in the paper.
  - **Dual bound for strictly convex QPs.** For a convex QP with
    ``Q \succ 0`` the Lagrangian is bounded below in ``x`` for every
    multiplier, so by weak duality the completed multipliers give a valid
    bound `L2OCore.dual_bound` ``\le`` OPT. This is a property of the problem
    class, not of the method.

**What the method does not provide.** Feasibility of the predicted ``x̂(p)``
(beyond variable bounds with a bound-enforcing map), optimality, and a valid
dual bound in general are not ensured: the predicted multipliers satisfy sign
conditions but not Lagrangian stationarity. Classical ALM convergence
(Andreani et al., 2008, on which the penalty rule is modelled) concerns exact
or approximate minimisation per instance; it does not transfer to stochastic
training of a finite network on sampled instances, and the paper does not
claim it does: it states that the iterates "are expected to converge to the
primal and dual solutions on the training instances" (§4.4), and proves no
theorem about PDL. Measure feasibility and optimality gaps with
`L2OCore.evaluate`.

## Deviations from the paper

  - **Box rows.** One signed multiplier per row instead of separate
    ``\mu``/``\lambda`` blocks; the formulas reduce to eqs. (5)–(8) as shown.
  - **Multiplier clip** ``M`` = `max_dual` (default ``10^6``), a numerical
    safeguard kept from the original implementation.
  - **Smoothed norms** in (B5) (``\varepsilon = 10^{-8}``), so the dual loss is
    differentiable when the prediction equals the target.
  - **Epochs.** Inner iterations ``L`` are counted in epochs over the training
    set (`primal_epochs`, `dual_epochs`), as in the paper's experiments.
  - **PHR option** (B3), not in the paper.
  - **Variable bounds without a bound map** are dualized like any other row;
    the paper always enforces ``\mathbf Y`` by the architecture (see above).
    Use a bound-enforcing map to match it.
  - **Learning-rate decay** of App. C.3 (×0.99 when the validation loss
    exceeds its best value) is not built into [`train!`](@ref); the `callback`
    keyword can adjust the optimiser between outer iterations.
  - **Norm in (7)**: Euclidean (the paper allows any norm).
  - **Eq. (8)**: ``\lambda_{k,j}`` for ``j \in \mathcal G`` read as
    ``\mu_{k,j}`` (see above).

## Related methods

  - Kotary & Fioretto, *Learning Constrained Optimization with Deep Augmented
    Lagrangian Methods*, 2024 ([arXiv:2403.03454](https://arxiv.org/abs/2403.03454)):
    a single network predicts equality multipliers ``\nu``; the primal point is
    recovered per instance by a box-constrained solve of the augmented
    Lagrangian, and ``\rho`` grows geometrically per epoch. It would be another
    `AbstractPrimalDualMethod` on the same interface (it needs a per-instance
    inner solver, so it does not reuse [`train!`](@ref)).
  - "Analytical dual" PDL for AC-OPF: use ``\mathrm{clip}(\lambdâ + \rho\,\sigma)``
    inside the primal loss. Its gradient is the one of (B3), so it is available
    as `augmentation = PHRAugmentation()`; see [Design](@ref).
