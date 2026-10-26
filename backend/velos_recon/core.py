# ---------------------------------------------------------------------------
# PROVENANCE: This file is the Client's (Velos BioImaging / M. Awais) existing
# Python physics-based reconstruction code, ptycho_core.py (MAP paper codes,
# K:\Samsung Work\Awais_Ptycho_2\MAP_TCI_Paper\Codes\ptycho_core.py).
# It is vendored UNCHANGED except for one integration shim marked
# "VELOS-INTEGRATION" inside _wrap(), which exposes the per-iteration step
# function so the backend can drive it with progress reporting. No update
# rule, gradient, metric or simulation logic has been modified
# (Contract Schedule A: validated scientific logic is not rewritten).
# ---------------------------------------------------------------------------
"""
ptycho_core.py
==============
Simulation framework and reconstruction engines for the MAP Optics Letters paper.

Engines (all share the same Wirtinger gradient, Eq. 2; they differ only in D):
    GD           D = 1                                          [batch]
    ePIE         D = max|P|^2                                   [sequential]
    rPIE         D = (1-a)|P|^2 + a*max|P|^2                    [sequential]
    WASP         D = sum_j |P(r-r_j)|^2                         [batch]
    Adam (naive) v_re, v_im tracked SEPARATELY                  [batch]
    MAP (ours)   D = sqrt(<|g|^2>), COMPLEX-COUPLED             [batch or stochastic]

KEY DESIGN POINT -- state this in the paper:
    MAP uses a COMPLEX-COUPLED second moment, v = |g|^2 = gr^2 + gi^2, a single
    REAL scalar per pixel. Stock Adam applied to [Re(O), Im(O)] as two real
    tensors tracks v_re and v_im independently, rescaling real and imaginary
    parts by DIFFERENT factors. That ROTATES the update direction in the complex
    plane and distorts phase. MAP's coupled moment preserves the Wirtinger
    direction and rescales only its magnitude.
    `adam_naive` implements the decoupled version so the comparison is honest.
"""

import numpy as np
import time


# ----------------------------------------------------------------------
# Fourier transforms (centred)
# ----------------------------------------------------------------------
def FT(x):
    n = x.shape[-1]
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1))),
                           axes=(-2, -1)) / n


def IFT(X):
    n = X.shape[-1]
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X, axes=(-2, -1))),
                           axes=(-2, -1)) * n


def _smooth(a, sigma):
    """Gaussian blur via FFT (avoids a scipy dependency)."""
    N = a.shape[0]
    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.fftfreq(N)[None, :]
    H = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * H))


# ----------------------------------------------------------------------
# Object / probe / scan geometry
# ----------------------------------------------------------------------
def make_object(N=256, seed=0):
    """Complex object: spoke-target amplitude x smooth structured phase."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(float)
    xc, yc = x - N / 2, y - N / 2
    r = np.hypot(xc, yc)
    th = np.arctan2(yc, xc)

    amp = 0.5 + 0.5 * np.sign(np.sin(16 * th))
    amp = amp * (r < 0.42 * N)
    amp = amp + 0.35 * (np.sin(r / 6.0) > 0.4) * (r < 0.46 * N)
    amp = np.clip(amp, 0, 1)
    amp = _smooth(amp, 1.2)
    amp = 0.25 + 0.75 * (amp - amp.min()) / (np.ptp(amp) + 1e-12)

    pha = (1.2 * np.exp(-(r / (0.30 * N)) ** 2)
           * np.sin(2 * np.pi * xc / (0.35 * N))
           * np.cos(2 * np.pi * yc / (0.45 * N)))
    pha = pha + 0.25 * _smooth(rng.standard_normal((N, N)), 6.0)

    return (amp * np.exp(1j * pha)).astype(np.complex128)


def make_probe(Np=64):
    """Soft-edged aperture with mild defocus -> structured probe."""
    u, v = np.meshgrid(np.linspace(-1, 1, Np), np.linspace(-1, 1, Np))
    r = np.hypot(u, v)
    ap = 1.0 / (1.0 + np.exp((r - 0.62) / 0.035))
    P = ap * np.exp(1j * 2.2 * r ** 2)
    P = P / np.abs(P).max()          # max|P| = 1  <-- balances object/probe update scales
    return P.astype(np.complex128)


def make_positions(N, Np, overlap=0.70, jitter=0.0, seed=0):
    """Raster grid at the requested linear overlap fraction."""
    rng = np.random.default_rng(seed)
    step = max(1, int(round(Np * (1.0 - overlap))))
    idx = np.arange(0, N - Np + 1, step)
    rr, cc = np.meshgrid(idx, idx, indexing='ij')
    pos = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
    if jitter > 0:
        pos = pos + rng.normal(0, jitter, pos.shape)
    return np.clip(np.round(pos), 0, N - Np).astype(int)


def compute_denO(P, pos, shape):
    """Accumulated illumination intensity = Gauss-Newton Hessian diagonal."""
    Np = P.shape[0]
    d = np.zeros(shape, dtype=np.float64)
    ip = np.abs(P) ** 2
    for (r0, c0) in pos:
        d[r0:r0 + Np, c0:c0 + Np] += ip
    return d


def simulate(O, P, pos, photons=1e4, seed=0):
    """Forward model + Poisson noise. Returns measured AMPLITUDES [J,Np,Np]."""
    rng = np.random.default_rng(seed)
    Np = P.shape[0]
    amps = np.zeros((len(pos), Np, Np))
    for k, (r0, c0) in enumerate(pos):
        I = np.abs(FT(P * O[r0:r0 + Np, c0:c0 + Np])) ** 2
        if photons is None or photons <= 0:
            amps[k] = np.sqrt(I)
        else:
            s = photons / I.mean()
            amps[k] = np.sqrt(np.maximum(rng.poisson(I * s) / s, 0.0))
    return amps


# ----------------------------------------------------------------------
# Metrics (complex scalar ambiguity removed first)
# ----------------------------------------------------------------------
def _align(O_rec, O_true, mask):
    a, b = O_rec[mask], O_true[mask]
    return np.vdot(a, b) / (np.vdot(a, a) + 1e-30)


def nrmse(O_rec, O_true, mask):
    c = _align(O_rec, O_true, mask)
    a, b = c * O_rec[mask], O_true[mask]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def ssim_amp(O_rec, O_true, mask):
    c = _align(O_rec, O_true, mask)
    x = np.abs(c * O_rec[mask]); y = np.abs(O_true[mask])
    x = x / (x.max() + 1e-30); y = y / (y.max() + 1e-30)
    mx, my, vx, vy = x.mean(), y.mean(), x.var(), y.var()
    vxy = float(((x - mx) * (y - my)).mean())
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mx * my + C1) * (2 * vxy + C2)) /
                 ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2)))


# ----------------------------------------------------------------------
# Shared gradient (Eq. 2)
# ----------------------------------------------------------------------
def _patch_grad(O, P, amps_k, r0, c0, Np):
    patch = O[r0:r0 + Np, c0:c0 + Np]
    psi = P * patch
    Psi = FT(psi)
    Psi_m = amps_k * np.exp(1j * np.angle(Psi))
    dpsi = IFT(Psi_m) - psi
    gO = np.conj(P) * dpsi
    gP = np.conj(patch) * dpsi
    err = float(np.sum((np.abs(Psi) - amps_k) ** 2))
    return gO, gP, err, patch


def batch_gradient(O, P, amps, pos):
    Np = P.shape[0]
    gO = np.zeros_like(O)
    gP = np.zeros_like(P)
    err = 0.0
    for k, (r0, c0) in enumerate(pos):
        go, gp, e, _ = _patch_grad(O, P, amps[k], r0, c0, Np)
        gO[r0:r0 + Np, c0:c0 + Np] += go
        gP += gp
        err += e
    return gO, gP, err


def cosine_lr(lr0, n, n_total, lr_min_frac=0.02):
    """Cosine annealing. Adam's step magnitude is ~lr regardless of gradient
    scale, so without decay it plateaus at a noise floor of order lr."""
    p = (n - 1) / max(n_total - 1, 1)
    lr_min = lr0 * lr_min_frac
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * p))


def _init(shape, P0, seed):
    rng = np.random.default_rng(seed + 999)
    O = np.ones(shape, dtype=np.complex128) * np.exp(
        0.3j * rng.standard_normal(shape))
    return O, P0.astype(np.complex128).copy()


def _wrap(step_fn):
    """Adds timing + per-iteration metric tracking to an engine step."""
    def run(amps, P0, pos, O_true, n_iter=100, seed=0, mask=None, **kw):
        N = O_true.shape[0]
        O, P = _init((N, N), P0, seed)
        if mask is None:
            d = compute_denO(P0, pos, (N, N))
            mask = d > 0.05 * d.max()
        hist = {'nrmse': [], 'ssim': [], 'time': [], 'err': []}
        state = {'n_total': n_iter}
        t0 = time.perf_counter()
        for n in range(1, n_iter + 1):
            O, P, err = step_fn(O, P, amps, pos, n, state, **kw)
            hist['time'].append(time.perf_counter() - t0)
            hist['err'].append(err)
            hist['nrmse'].append(nrmse(O, O_true, mask))
            hist['ssim'].append(ssim_amp(O, O_true, mask))
        out = {k: np.asarray(v) for k, v in hist.items()}
        out['state'] = state
        return O, P, out
    run.step_fn = step_fn   # VELOS-INTEGRATION: expose the validated step function
    return run


# ======================================================================
#  ENGINES
# ======================================================================
@_wrap
def gd(O, P, amps, pos, n, state, probe_lr=0.3):
    """D = 1. Step fixed by inverse max curvature."""
    if 'L' not in state:
        state['L'] = compute_denO(P, pos, O.shape).max()
    gO, gP, err = batch_gradient(O, P, amps, pos)
    O = O + gO / (state['L'] + 1e-12)
    denP = sum(np.abs(O[r:r + P.shape[0], c:c + P.shape[0]]) ** 2
               for r, c in pos)
    P = P + probe_lr * gP / (denP.max() + 1e-12)
    return O, P, err


@_wrap
def epie(O, P, amps, pos, n, state, alpha=0.5, beta=0.5):
    """D = max|P|^2, sequential."""
    Np = P.shape[0]
    rng = np.random.default_rng(n * 7919)
    err = 0.0
    for k in rng.permutation(len(pos)):
        r0, c0 = pos[k]
        gO, gP, e, patch = _patch_grad(O, P, amps[k], r0, c0, Np)
        err += e
        O[r0:r0 + Np, c0:c0 + Np] += alpha * gO / (np.abs(P).max() ** 2 + 1e-12)
        P += beta * gP / (np.abs(patch).max() ** 2 + 1e-12)
    return O, P, err


@_wrap
def rpie(O, P, amps, pos, n, state, a_obj=0.15, a_prb=0.15, beta=0.5):
    """D = (1-a)|P|^2 + a*max|P|^2, sequential."""
    Np = P.shape[0]
    rng = np.random.default_rng(n * 7919)
    err = 0.0
    for k in rng.permutation(len(pos)):
        r0, c0 = pos[k]
        gO, gP, e, patch = _patch_grad(O, P, amps[k], r0, c0, Np)
        err += e
        dP = (1 - a_obj) * np.abs(P) ** 2 + a_obj * np.abs(P).max() ** 2
        O[r0:r0 + Np, c0:c0 + Np] += gO / (dP + 1e-12)
        dO = (1 - a_prb) * np.abs(patch) ** 2 + a_prb * np.abs(patch).max() ** 2
        P += beta * gP / (dO + 1e-12)
    return O, P, err


@_wrap
def wasp(O, P, amps, pos, n, state, probe_lr=1.0):
    """D = sum_j |P(r-r_j)|^2, batch."""
    Np = P.shape[0]
    gO, gP, err = batch_gradient(O, P, amps, pos)
    denO = compute_denO(P, pos, O.shape)
    denP = np.zeros(P.shape)
    for (r0, c0) in pos:
        denP += np.abs(O[r0:r0 + Np, c0:c0 + Np]) ** 2
    O = O + gO / (denO + 1e-9)
    P = P + probe_lr * gP / (denP + 1e-9)
    return O, P, err


@_wrap
def adam_naive(O, P, amps, pos, n, state,
               lr=1.5e-2, b1=0.9, b2=0.999, eps=1e-8, probe_lr=1.5e-3,
               anneal=True):
    """
    Stock Adam on [Re(O), Im(O)] as SEPARATE real parameters.

    IDENTICAL to map_engine in every respect EXCEPT that v_re and v_im are
    tracked independently.  Real and imaginary parts are therefore rescaled by
    DIFFERENT factors, which ROTATES the complex update direction and distorts
    phase.  This isolates the contribution of MAP's complex-coupled moment.
    """
    Np = P.shape[0]
    if 'm' not in state:
        state['m'] = np.zeros(O.shape + (2,))
        state['v'] = np.zeros(O.shape + (2,))
        state['mP'] = np.zeros(P.shape + (2,))
        state['vP'] = np.zeros(P.shape + (2,))
        state['t'] = 0
    if anneal:
        lr = cosine_lr(lr, n, state['n_total'])
        probe_lr = cosine_lr(probe_lr, n, state['n_total'])

    rng = np.random.default_rng(n * 7919)
    err = 0.0
    for k in rng.permutation(len(pos)):
        r0, c0 = pos[k]
        go, gp, e, _ = _patch_grad(O, P, amps[k], r0, c0, Np)
        err += e
        state['t'] += 1
        t = state['t']

        gf = np.zeros_like(O); gf[r0:r0 + Np, c0:c0 + Np] = go
        g2 = np.stack([gf.real, gf.imag], -1)
        state['m'] = b1 * state['m'] + (1 - b1) * g2
        state['v'] = b2 * state['v'] + (1 - b2) * g2 ** 2        # DECOUPLED
        mh = state['m'] / (1 - b1 ** t)
        vh = state['v'] / (1 - b2 ** t)
        s = lr * mh / (np.sqrt(vh) + eps)
        O = O + (s[..., 0] + 1j * s[..., 1])

        p2 = np.stack([gp.real, gp.imag], -1)
        state['mP'] = b1 * state['mP'] + (1 - b1) * p2
        state['vP'] = b2 * state['vP'] + (1 - b2) * p2 ** 2
        mhP = state['mP'] / (1 - b1 ** t)
        vhP = state['vP'] / (1 - b2 ** t)
        sp = probe_lr * mhP / (np.sqrt(vhP) + eps)
        P = P + (sp[..., 0] + 1j * sp[..., 1])
    return O, P, err


@_wrap
def map_engine(O, P, amps, pos, n, state,
               lr=1.5e-2, b1=0.9, b2=0.999, eps=1e-8, probe_lr=1.5e-3,
               mode='stochastic', anneal=True):
    """
    MAP.  Eq. (3): m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*|g|^2   [COUPLED]
          Eq. (4): O <- O + alpha * m_hat / (sqrt(v_hat) + eps)

    v is a single REAL array, so the complex direction of m_hat is preserved
    and only its magnitude is rescaled.
    """
    Np = P.shape[0]
    if 'm' not in state:
        state['m'] = np.zeros(O.shape, dtype=np.complex128)
        state['v'] = np.zeros(O.shape, dtype=np.float64)
        state['mP'] = np.zeros(P.shape, dtype=np.complex128)
        state['vP'] = np.zeros(P.shape, dtype=np.float64)
        state['t'] = 0
    if anneal:
        lr = cosine_lr(lr, n, state['n_total'])
        probe_lr = cosine_lr(probe_lr, n, state['n_total'])

    if mode == 'batch':
        gO, gP, err = batch_gradient(O, P, amps, pos)
        state['t'] += 1
        t = state['t']
        state['m'] = b1 * state['m'] + (1 - b1) * gO
        state['v'] = b2 * state['v'] + (1 - b2) * np.abs(gO) ** 2   # COUPLED
        mh = state['m'] / (1 - b1 ** t)
        vh = state['v'] / (1 - b2 ** t)
        O = O + lr * mh / (np.sqrt(vh) + eps)

        state['mP'] = b1 * state['mP'] + (1 - b1) * gP
        state['vP'] = b2 * state['vP'] + (1 - b2) * np.abs(gP) ** 2
        mhP = state['mP'] / (1 - b1 ** t)
        vhP = state['vP'] / (1 - b2 ** t)
        P = P + probe_lr * mhP / (np.sqrt(vhP) + eps)
    else:
        rng = np.random.default_rng(n * 7919)
        err = 0.0
        for k in rng.permutation(len(pos)):
            r0, c0 = pos[k]
            go, gp, e, _ = _patch_grad(O, P, amps[k], r0, c0, Np)
            err += e
            state['t'] += 1
            t = state['t']
            gf = np.zeros_like(O); gf[r0:r0 + Np, c0:c0 + Np] = go
            state['m'] = b1 * state['m'] + (1 - b1) * gf
            state['v'] = b2 * state['v'] + (1 - b2) * np.abs(gf) ** 2
            mh = state['m'] / (1 - b1 ** t)
            vh = state['v'] / (1 - b2 ** t)
            O = O + lr * mh / (np.sqrt(vh) + eps)

            state['mP'] = b1 * state['mP'] + (1 - b1) * gp
            state['vP'] = b2 * state['vP'] + (1 - b2) * np.abs(gp) ** 2
            mhP = state['mP'] / (1 - b1 ** t)
            vhP = state['vP'] / (1 - b2 ** t)
            P = P + probe_lr * mhP / (np.sqrt(vhP) + eps)

    t = max(state['t'], 1)
    state['v_hat'] = state['v'] / (1 - b2 ** t)     # for Fig. 1(c) and Fig. 3
    state['m_hat'] = state['m'] / (1 - b1 ** t)
    return O, P, err


# ----------------------------------------------------------------------
ENGINES = {'GD': gd, 'ePIE': epie, 'rPIE': rpie, 'WASP': wasp,
           'Adam (naive)': adam_naive, 'MAP': map_engine}

COLORS = {'GD': '#94a3b8', 'ePIE': '#f59e0b', 'rPIE': '#10b981',
          'WASP': '#2563eb', 'Adam (naive)': '#a855f7', 'MAP': '#dc2626'}

STYLES = {'GD': (0, (1, 2)), 'ePIE': (0, (5, 2)), 'rPIE': (0, (4, 1, 1, 1)),
          'WASP': (0, (6, 2)), 'Adam (naive)': (0, (1, 1)), 'MAP': '-'}

ORDER = ['GD', 'ePIE', 'rPIE', 'WASP', 'Adam (naive)', 'MAP']
