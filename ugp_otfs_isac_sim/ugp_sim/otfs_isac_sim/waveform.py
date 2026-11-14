"""
OTFS and OFDM (non-OTFS) waveform layer.

Two levels:
  1. Link-level: ISFFT/SFFT, Heisenberg/Wigner with rectangular pulses, a
     delay-Doppler channel, and an empirical SINR measurement. Used to validate
     the abstraction in (2) and for the DD input-output check (Draft eq. y_k[k,l]).
  2. System-level abstraction used by the optimiser:
       - OTFS: integer Doppler is a cyclic shift in the DD grid -> no SINR loss
         (Draft Sec. V, assumption nu_k = k_k/(NT), l_k = 0).
       - OFDM: Doppler eps = nu/delta_f destroys subcarrier orthogonality.
         With rectangular pulses the desired-tap power fraction is ~sinc^2(eps)
         and the rest leaks as ICI (energy conservation).
     plus a rate prefactor for cyclic-prefix overhead:
       - OTFS: one CP of l_max samples per frame  -> MN/(MN + l_max)
       - OFDM: one CP of l_max samples per symbol -> M/(M + l_max)
"""
from __future__ import annotations

import functools

import numpy as np

from .config import SimParams

# --------------------------------------------------------------------------
# 1. Link level
# --------------------------------------------------------------------------


def isfft(x_dd: np.ndarray) -> np.ndarray:
    """x[k, l] (N x M, Doppler x delay) -> X[n, m] (N x M, time x freq).
    X[n,m] = 1/sqrt(MN) sum_k sum_l x[k,l] exp(j2pi(nk/N - ml/M))   (Draft Sec. II)
    """
    return np.fft.fft(np.fft.ifft(x_dd, axis=0, norm="ortho"), axis=1, norm="ortho")


def sfft(X_tf: np.ndarray) -> np.ndarray:
    """Inverse of isfft."""
    return np.fft.ifft(np.fft.fft(X_tf, axis=0, norm="ortho"), axis=1, norm="ortho")


def heisenberg(X_tf: np.ndarray) -> np.ndarray:
    """Rectangular-pulse multicarrier modulator: N symbols of M samples, no per-symbol CP.
    Returns time samples of length N*M at fs = M*delta_f."""
    return np.fft.ifft(X_tf, axis=1, norm="ortho").reshape(-1)


def wigner(r: np.ndarray, N: int, M: int) -> np.ndarray:
    return np.fft.fft(r.reshape(N, M), axis=1, norm="ortho")


def otfs_modulate(x_dd: np.ndarray) -> np.ndarray:
    return heisenberg(isfft(x_dd))


def otfs_demodulate(r: np.ndarray, N: int, M: int) -> np.ndarray:
    return sfft(wigner(r, N, M))


def ofdm_modulate(X_tf: np.ndarray) -> np.ndarray:
    return heisenberg(X_tf)


def ofdm_demodulate(r: np.ndarray, N: int, M: int) -> np.ndarray:
    return wigner(r, N, M)


def dd_channel(s: np.ndarray, gain: complex, delay_samples: int, nu: float,
               fs: float) -> np.ndarray:
    """Single-path delay-Doppler channel. Frame-level cyclic prefix assumed, so the
    delay is circular over the whole frame (OTFS reduced-CP convention)."""
    n = np.arange(s.size)
    return gain * np.roll(s, delay_samples) * np.exp(1j * 2 * np.pi * nu * n / fs)


def qpsk(shape, rng: np.random.Generator) -> np.ndarray:
    return ((rng.integers(0, 2, shape) * 2 - 1) + 1j * (rng.integers(0, 2, shape) * 2 - 1)) / np.sqrt(2)


@functools.lru_cache(maxsize=16)
def _probe_waveform(N: int, M: int, seed: int) -> np.ndarray:
    s = otfs_modulate(qpsk((N, M), np.random.default_rng(seed)))
    s.setflags(write=False)
    return s


def bandlimited_shift(s: np.ndarray, fs: float, gamma: float) -> np.ndarray:
    """s(t - gamma) on the sample grid, for one period of a band-limited signal
    (fractional delays by a frequency-domain phase ramp; Nyquist bin split symmetrically)."""
    L = s.size
    S = np.fft.fft(s)
    f = np.fft.fftfreq(L, 1.0 / fs)
    ph = np.exp(-1j * 2 * np.pi * f * gamma)
    if L % 2 == 0:
        ph[L // 2] = np.cos(np.pi * fs * gamma)
    return np.fft.ifft(S * ph)


def probe_waveform(p: SimParams, seed: int = 0) -> np.ndarray:
    """The sensing waveform s0(t) used for the CRB (Draft Sec. III / Appendix A):
    one OTFS frame with unit-energy-per-symbol QPSK in the DD domain, rectangular
    pulses, N*M complex samples at fs = M*delta_f, duration N*T. sum|s|^2 = M*N
    (= Gm, the matched-filter gain). The ortho-normalised (ISFFT, Heisenberg) pair
    preserves energy, so the mean sample power is 1.

    The samples are read as ONE PERIOD of a band-limited signal occupying
    [-fs/2, fs/2) (frame-level CP convention of `dd_channel`); the tone at m = M/2
    sits on the band edge and is split symmetrically between -fs/2 and +fs/2."""
    return _probe_waveform(p.N, p.M, seed)


def measure_ofdm_sinr_loss(nu: float, p: SimParams, rng=None) -> float:
    """Empirical fraction of power left on the desired subcarrier tap after a
    Doppler shift nu (noise-free, single tap equalisation, known channel)."""
    rng = rng or np.random.default_rng(0)
    X = qpsk((p.N, p.M), rng)
    r = dd_channel(ofdm_modulate(X), 1.0, 0, nu, p.fs)
    Y = ofdm_demodulate(r, p.N, p.M)
    # per-symbol common phase is equalisable; measure coherent desired power
    h_hat = np.mean(Y / X, axis=1, keepdims=True)
    desired = np.abs(h_hat) ** 2
    err = np.mean(np.abs(Y - h_hat * X) ** 2, axis=1, keepdims=True)
    return float(np.mean(desired / (desired + err)))


def check_otfs_dd_shift(p: SimParams, k_nu: int, rng=None) -> float:
    """Verify Draft eq.: y[k,l] = h x[(k-k_nu)_N, l] for integer Doppler, l_tau = 0.

    With rectangular pulses the exact relation carries an extra deterministic
    phase exp(j 2 pi l k_nu / (MN)) (Raviteja et al. 2018) from the Doppler
    rotation inside each symbol. It costs no power and is known once k_nu is
    predicted, so it is compensable -- but the Draft's DD equation omits it.
    Returns max abs error (should be ~1e-14)."""
    rng = rng or np.random.default_rng(1)
    x = qpsk((p.N, p.M), rng)
    nu = k_nu / p.frame_duration
    y = otfs_demodulate(dd_channel(otfs_modulate(x), 1.0, 0, nu, p.fs), p.N, p.M)
    phase = np.exp(1j * 2 * np.pi * np.arange(p.M) * k_nu / (p.M * p.N))[None, :]
    return float(np.max(np.abs(y - np.roll(x, k_nu, axis=0) * phase)))


# --------------------------------------------------------------------------
# 2. System-level abstraction
# --------------------------------------------------------------------------


def doppler_power_split(nu, waveform: str, p: SimParams):
    """Return (kept_fraction, leaked_fraction) of received power per stream."""
    nu = np.asarray(nu, dtype=float)
    if waveform == "otfs":
        return np.ones_like(nu), np.zeros_like(nu)
    if waveform == "ofdm":
        kept = np.sinc(nu / p.delta_f) ** 2
        return kept, 1.0 - kept
    raise ValueError(waveform)


def rate_prefactor(waveform: str, p: SimParams) -> float:
    if waveform == "otfs":
        return p.M * p.N / (p.M * p.N + p.l_max)
    if waveform == "ofdm":
        return p.M / (p.M + p.l_max)
    raise ValueError(waveform)


def effective_sinr(signal, interference, noise, nu, waveform: str, p: SimParams):
    """SINR after the waveform-dependent Doppler impairment.
    signal / interference: received powers before impairment."""
    kept, leak = doppler_power_split(nu, waveform, p)
    return signal * kept / (interference * kept + leak * (signal + interference) + noise)
