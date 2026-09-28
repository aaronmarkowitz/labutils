"""Scattered-light -> size model for the particle loader (numpy only).

Bohren & Huffman Mie theory for a homogeneous sphere, integrated over the camera's
collection cone about 90 deg from the illuminating beam. Only RATIOS are ever
used: the absolute collected power depends on the lens, the camera QE and gain,
and the beam waist, none of which we know well. The ledger's reference particles
(size from an independent measurement, e.g. Epstein mass) fix the scale, and the
model supplies the shape of brightness vs diameter.

Geometry: beam along +z, linearly polarised at angle `pol_deg` from the
camera axis's azimuth, camera looking along an axis at 90 deg to the beam. With
the polarisation unknown the band over pol in {0, 90 deg} is reported.

Caveats (stated in every estimate): a cluster of nanodiamonds is not a sphere and
scatters less per unit mass than a solid sphere of the same mass; below ~lambda/3
brightness ~ d^6 (Rayleigh), so the size estimate is far better than the
brightness estimate there; above ~lambda the Mie ripple makes the curve
non-monotonic and a brightness can map to several diameters -- all are reported.

    python3 size_model.py --self-test
"""
import math
import sys

import numpy as np

N_DIAMOND_980 = 2.40      # real index of diamond near 980 nm (~2.39-2.41 across the NIR)
RHO_DIAMOND = 3515.0      # kg/m^3


def diameter_from_mass(mass_kg, rho=RHO_DIAMOND):
    """Equivalent-sphere diameter (m) of a solid sphere of mass `mass_kg`."""
    return (6.0 * mass_kg / (math.pi * rho)) ** (1.0 / 3.0)


def mass_from_diameter(d_m, rho=RHO_DIAMOND):
    return rho * math.pi * d_m ** 3 / 6.0


def mie_coefficients(m, x):
    """Mie a_n, b_n (n = 1..nmax) for relative index m and size parameter x.

    Bohren & Huffman BHMIE: logarithmic derivative D_n by downward recurrence,
    Riccati-Bessel psi/xi by upward recurrence.
    """
    m = complex(m)
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    y = m * x
    nmx = int(max(nstop, abs(y))) + 15
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / y - 1.0 / (D[n] + n / y)
    psi0, psi1 = math.cos(x), math.sin(x)
    chi0, chi1 = -math.sin(x), math.cos(x)
    xi1 = complex(psi1, -chi1)
    a = np.zeros(nstop, dtype=complex)
    b = np.zeros(nstop, dtype=complex)
    for n in range(1, nstop + 1):
        psi = (2.0 * n - 1.0) * psi1 / x - psi0
        chi = (2.0 * n - 1.0) * chi1 / x - chi0
        xi = complex(psi, -chi)
        a[n - 1] = ((D[n] / m + n / x) * psi - psi1) / ((D[n] / m + n / x) * xi - xi1)
        b[n - 1] = ((D[n] * m + n / x) * psi - psi1) / ((D[n] * m + n / x) * xi - xi1)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
    return a, b


def efficiencies(m, x):
    """(Q_ext, Q_sca, Q_back) -- used by the self-test against published values."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, len(a) + 1)
    qext = 2.0 / x ** 2 * np.sum((2 * n + 1) * (a + b).real)
    qsca = 2.0 / x ** 2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))
    qback = abs(np.sum((2 * n + 1) * (-1.0) ** n * (a - b))) ** 2 / x ** 2
    return float(qext), float(qsca), float(qback)


def amplitude_functions(m, x, theta):
    """S1(theta), S2(theta) for an array of scattering angles (rad)."""
    a, b = mie_coefficients(m, x)
    mu = np.cos(np.atleast_1d(theta))
    S1 = np.zeros(mu.shape, dtype=complex)
    S2 = np.zeros(mu.shape, dtype=complex)
    pi0 = np.zeros_like(mu)
    pi1 = np.ones_like(mu)
    for n in range(1, len(a) + 1):
        tau = n * mu * pi1 - (n + 1) * pi0
        f = (2.0 * n + 1.0) / (n * (n + 1.0))
        S1 += f * (a[n - 1] * pi1 + b[n - 1] * tau)
        S2 += f * (a[n - 1] * tau + b[n - 1] * pi1)
        pi0, pi1 = pi1, ((2.0 * n + 1.0) * mu * pi1 - (n + 1.0) * pi0) / n
    return S1, S2


def _cone_directions(na, n_ring=12, n_az=24):
    """Unit vectors and solid-angle weights filling a cone of half-angle asin(na)
    about the +x axis (the camera axis; beam along +z)."""
    alpha = math.asin(min(max(na, 1e-6), 0.999))
    # Gauss-Legendre in cos(psi) over [cos(alpha), 1], uniform in azimuth.
    u, w = np.polynomial.legendre.leggauss(n_ring)
    c0 = math.cos(alpha)
    cpsi = 0.5 * (1 - c0) * u + 0.5 * (1 + c0)
    wpsi = 0.5 * (1 - c0) * w
    az = np.linspace(0, 2 * math.pi, n_az, endpoint=False)
    C, A = np.meshgrid(cpsi, az, indexing='ij')
    W = np.repeat(wpsi[:, None], n_az, axis=1) * (2 * math.pi / n_az)
    S = np.sqrt(1 - C ** 2)
    # camera axis x; cone spans y,z around it
    vx, vy, vz = C, S * np.cos(A), S * np.sin(A)
    return vx.ravel(), vy.ravel(), vz.ravel(), W.ravel()


def collected_signal(d_m, wavelength_m=980e-9, n_particle=N_DIAMOND_980, n_medium=1.0,
                     na=0.1, pol_deg=90.0):
    """Relative power scattered into the camera's collection cone.

    Units: (lambda/2pi)^2 * integral |S|^2 dOmega in m^2 -- i.e. a partial
    scattering cross-section. Only ratios between diameters are meaningful for
    the loader, but the absolute value is a real cross-section, which the
    self-test checks against Q_sca in the full-sphere limit.

    pol_deg: incident polarisation angle measured from the camera axis (x) in the
    x-y plane. 90 = polarisation perpendicular to the viewing direction (dipole
    radiates maximally toward the camera); 0 = along it (Rayleigh signal
    vanishes on axis).
    """
    k = 2 * math.pi * n_medium / wavelength_m
    x = k * d_m / 2
    m = n_particle / n_medium
    vx, vy, vz, W = _cone_directions(na)
    theta = np.arccos(np.clip(vz, -1, 1))
    phi = np.arctan2(vy, vx)                     # azimuth of scattering direction
    S1, S2 = amplitude_functions(m, x, theta)
    p = math.radians(pol_deg)
    dphi = phi - p                                # azimuth relative to polarisation
    I = abs(S2) ** 2 * np.cos(dphi) ** 2 + abs(S1) ** 2 * np.sin(dphi) ** 2
    return float(np.sum(I * W) / k ** 2)


def signal_curve(diameters_m, **kw):
    return np.array([collected_signal(d, **kw) for d in diameters_m])


DEFAULT_GRID_M = np.geomspace(20e-9, 6e-6, 700)


def estimate_diameter(brightness, refs, grid_m=DEFAULT_GRID_M, rel_tol=0.5,
                      pols=(0.0, 45.0, 90.0), **kw):
    """Diameters consistent with `brightness`, given reference particles.

    refs: list of (brightness_ref, d_ref_m). The model scale for each polarisation
    is the geometric mean of B_ref / C(d_ref) over refs. A diameter is consistent
    when |log(B / (scale * C(d)))| < log(1 + rel_tol) for ANY polarisation in
    `pols`. Returns dict with the candidate range, the central estimate at the
    middle polarisation, and whether the curve was multi-valued.
    """
    if not refs:
        raise ValueError("need at least one reference particle to set the scale")
    lo, hi = math.inf, -math.inf
    central = None
    multi = False
    for pol in pols:
        C = signal_curve(grid_m, pol_deg=pol, **kw)
        logscale = np.mean([math.log(Br / collected_signal(dr, pol_deg=pol, **kw))
                            for Br, dr in refs])
        pred = logscale + np.log(C)
        ok = np.abs(math.log(brightness) - pred) < math.log(1 + rel_tol)
        if ok.any():
            lo = min(lo, grid_m[ok].min())
            hi = max(hi, grid_m[ok].max())
            runs = np.count_nonzero(np.diff(ok.astype(int)) == 1) + int(ok[0])
            multi = multi or runs > 1
        if pol == pols[len(pols) // 2]:
            central = float(grid_m[np.argmin(np.abs(math.log(brightness) - pred))])
    if not math.isfinite(lo):
        # Outside the modelled range: report the nearest grid end
        lo = hi = central
    return {'d_central_m': central, 'd_min_m': float(lo), 'd_max_m': float(hi),
            'multivalued': bool(multi)}


def predicted_ratio(d_m, d_ref_m, pol_deg=90.0, **kw):
    """Brightness of a d_m sphere relative to a d_ref_m sphere."""
    return collected_signal(d_m, pol_deg=pol_deg, **kw) / collected_signal(d_ref_m, pol_deg=pol_deg, **kw)


# --------------------------------------------------------------------------- self-test

def _checks(m_golden=1.55):
    """Return list of (name, passed, detail). `m_golden` is exposed so the
    control can feed a wrong index and prove the golden check fails."""
    out = []
    # 1. Golden value: Bohren & Huffman Appendix A example, m=1.55, x=5.213
    #    (lambda=0.6328 um, r=0.525 um): Qsca = Qext = 3.1054, Qback = 2.9253.
    qe, qs, qb = efficiencies(m_golden, 5.213)
    ok = abs(qs - 3.1054) < 2e-3 and abs(qe - 3.1054) < 2e-3 and abs(qb - 2.9253) < 5e-3
    out.append(('BH golden m=1.55 x=5.213', ok, f'Qext={qe:.4f} Qsca={qs:.4f} Qback={qb:.4f}'))
    # 2. Rayleigh limit: Qsca -> 8/3 x^4 |(m^2-1)/(m^2+2)|^2
    x, m = 0.01, 2.40
    _, qs, _ = efficiencies(m, x)
    qr = 8.0 / 3.0 * x ** 4 * abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2
    out.append(('Rayleigh limit Qsca', abs(qs / qr - 1) < 1e-3, f'ratio={qs / qr:.6f}'))
    # 3. Rayleigh angular: S2/S1 = cos(theta)
    S1, S2 = amplitude_functions(m, x, np.array([0.3, 1.0, 2.0]))
    r = (S2 / S1).real / np.cos([0.3, 1.0, 2.0])
    out.append(('Rayleigh S2/S1=cos', np.allclose(r, 1, atol=1e-4), f'{r}'))
    # 4. Extinction paradox: Qext -> 2 from above for large x, with the van de
    #    Hulst edge term ~2 x^(-2/3) (= +0.037 at x=400), so allow 0 < Qext-2 < 0.05.
    qe, _, _ = efficiencies(1.33, 400.0)
    out.append(('Qext -> 2 at x=400', 0 < qe - 2 < 0.05, f'Qext={qe:.4f}'))
    # 5. Small-particle d^6 scaling of the collected signal
    c1 = collected_signal(20e-9)
    c2 = collected_signal(40e-9)
    out.append(('Collected ~ d^6 (Rayleigh)', abs(c2 / c1 / 64 - 1) < 0.01, f'{c2 / c1:.3f} vs 64'))
    # 6. Rayleigh polarisation null: pol along camera axis -> near-zero at small NA
    c_par = collected_signal(20e-9, pol_deg=0.0, na=0.05)
    c_perp = collected_signal(20e-9, pol_deg=90.0, na=0.05)
    out.append(('Rayleigh pol null on axis', c_par / c_perp < 0.01, f'ratio={c_par / c_perp:.2e}'))
    # 7. Collected cross-section at small NA equals dCsca/dOmega * Omega
    d = 20e-9
    k = 2 * math.pi / 980e-9
    a = d / 2
    alpha_term = abs((2.4 ** 2 - 1) / (2.4 ** 2 + 2)) ** 2
    dcs = k ** 4 * a ** 6 * alpha_term                     # perp. polarisation, 90 deg
    na = 0.05
    omega = 2 * math.pi * (1 - math.cos(math.asin(na)))
    got = collected_signal(d, na=na, pol_deg=90.0)
    out.append(('Absolute dCsca/dOmega*Omega', abs(got / (dcs * omega) - 1) < 0.01,
                f'ratio={got / (dcs * omega):.4f}'))
    # 8. Inversion round trip in the monotonic regime
    refs = [(1.0, 1.76e-6)]
    for d_true in (100e-9, 300e-9):
        B = predicted_ratio(d_true, 1.76e-6, pol_deg=45.0)
        est = estimate_diameter(B, refs, rel_tol=0.05)
        ok = est['d_min_m'] <= d_true * 1.02 and est['d_max_m'] >= d_true * 0.98
        out.append((f'Inversion round trip {d_true * 1e9:.0f} nm', ok,
                    f"[{est['d_min_m'] * 1e9:.0f}, {est['d_max_m'] * 1e9:.0f}] nm"))
    return out


def self_test():
    ok_all = True
    for name, ok, detail in _checks():
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        ok_all &= bool(ok)
    # Control: the golden check must FAIL with a wrong refractive index.
    ctrl = _checks(m_golden=1.50)[0]
    ctrl_ok = not ctrl[1]
    print(f"{'PASS' if ctrl_ok else 'FAIL'}  control (golden check rejects m=1.50): {ctrl[2]}")
    ok_all &= ctrl_ok
    print('SELF-TEST', 'PASSED' if ok_all else 'FAILED')
    return ok_all


def _table():
    d_ref = diameter_from_mass(10e-15)
    print(f"10 pg diamond sphere -> d = {d_ref * 1e6:.3f} um")
    print(f"{'d (nm)':>8} {'mass (fg)':>10} " + ' '.join(f'{"B/Bref pol" + str(p):>14}' for p in (0, 45, 90)))
    for d in (50e-9, 100e-9, 200e-9, 300e-9, 500e-9, 800e-9, 1e-6, 1.76e-6, 3e-6):
        rs = [predicted_ratio(d, d_ref, pol_deg=p) for p in (0, 45, 90)]
        print(f"{d * 1e9:8.0f} {mass_from_diameter(d) * 1e18:10.3g} " + ' '.join(f'{r:14.3e}' for r in rs))


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        sys.exit(0 if self_test() else 1)
    _table()
