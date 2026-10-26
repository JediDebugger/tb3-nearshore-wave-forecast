"""Forward-marching (in x) half-plane spectral wave model with refraction, shoaling,
depth-induced breaking (Battjes-Janssen type), wave setup, crest height and rip-current
parameterisations.  All parameters live in a dict P (see DEFAULT_P / params.json)."""
import numpy as np

G = 9.81
DF = 0.02


def wavenumber(sig, h):
    """Linear dispersion sig^2 = g k tanh(k h), vectorised (h > 0)."""
    k0h = sig * sig * h / G
    # Guo (2002) explicit approximation, then two Newton iterations
    kh = k0h * (1.0 - np.exp(-np.power(k0h, 1.25))) ** (-0.4)
    for _ in range(2):
        t = np.tanh(kh)
        f = kh * t - k0h
        df = t + kh * (1.0 - t * t)
        kh = kh - f / df
    return kh / h


def spread(theta, mdir, s):
    d = np.abs(np.cos(0.5 * (theta - mdir))) ** (2.0 * s)
    return d


def boundary_spectrum(cond, freq, theta, P):
    """E(f, theta) [m^2/Hz/rad] at x = 0 on the half plane, from buoy-format input."""
    Ef = np.asarray(cond["Ef"], float)
    dirf = np.radians(np.asarray(cond["dirf"], float))
    sf = np.asarray(cond["sf"], float)
    thf = np.radians(np.arange(-180, 180, 0.25) + 0.125)
    dthf = np.radians(0.25)
    E = np.zeros((len(freq), len(theta)))
    dth = theta[1] - theta[0]
    for i in range(len(freq)):
        if Ef[i] <= 0:
            continue
        Dd = spread(thf, dirf[i], sf[i])
        Dd /= Dd.sum() * dthf
        # bin-average onto the model direction bins
        for j, t in enumerate(theta):
            m = np.abs(thf - t) <= 0.5 * dth + 1e-12
            E[i, j] = Ef[i] * Dd[m].mean()
    return E


def run(cond, depth_msl, x, y, freq, P, wet_struct=None, mwl=None, current_u=None, land=None):
    """March the spectrum from x=0 onshore.

    depth_msl: (nx, ny) depth below MSL on the product grid.
    mwl: (nx, ny) mean water level to use (if None -> tide).
    Returns dict of fields on (nx, ny)."""
    nx, ny = depth_msl.shape
    nsub = int(P.get("nsub", 2))
    dth_deg = P.get("dth", 5.0)
    theta = np.radians(np.arange(-90 + dth_deg / 2, 90, dth_deg))
    nth = len(theta)
    dth = np.radians(dth_deg)
    nf = len(freq)
    sig = 2 * np.pi * freq
    tide = cond["tide"]
    if mwl is None:
        mwl = np.full((nx, ny), tide)
    hmin = P.get("hmin", 0.05)
    E0 = boundary_spectrum(cond, freq, theta, P)  # (nf, nth)
    dxg = x[1] - x[0]
    dy = y[1] - y[0]
    dx = dxg / nsub
    cth = np.cos(theta)
    sth = np.sin(theta)
    tth = np.tan(theta)
    gam = P["gamma"]
    alpha = P["alpha"]
    nexp = P.get("fexp", 0.0)
    gam_k = P.get("gamma_k", 0.0)

    out = {k: np.full((nx, ny), np.nan) for k in ["Hs", "Tm", "Dm", "Qb", "Sxx", "Dis", "Hrms", "fm"]}
    # total depth on product grid
    htot = depth_msl + mwl
    if wet_struct is not None:
        htot = np.where(wet_struct, htot, -1.0)

    href = htot.copy()
    if land is not None and land.any():
        # depth used for refraction gradients: land cells take the depth of the nearest wet column
        for ii in range(nx):
            lj = np.where(land[ii])[0]
            wj = np.where(~land[ii])[0]
            if len(lj) and len(wj):
                near = wj[np.argmin(np.abs(wj[None, :] - lj[:, None]), axis=1)]
                href[ii, lj] = htot[ii, near]

    def depth_at(xx):
        # linear interpolation of htot and its gradients at continuous x
        fi = xx / dxg
        i0 = int(np.clip(np.floor(fi), 0, nx - 2))
        w = fi - i0
        h = (1 - w) * htot[i0] + w * htot[i0 + 1]
        hx = (htot[i0 + 1] - htot[i0]) / dxg
        hrow = (1 - w) * href[i0] + w * href[i0 + 1]
        hy = np.gradient(hrow, dy)
        return h, hx, hy

    # state: energy flux F = E Cg cos(theta)  (ny, nf, nth)
    h, hx, hy = depth_at(0.0)
    he = np.maximum(h, hmin)
    k = wavenumber(sig[None, :], he[:, None])  # (ny, nf)
    kh = k * he[:, None]
    n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
    cg = n * sig[None, :] / k
    F = E0[None, :, :] * cg[:, :, None] * cth[None, None, :]
    alive = np.ones(ny, bool)

    yy = y
    def diagnostics(F, cg, k, he, n, i):
        E = F / (cg[:, :, None] * cth[None, None, :])  # (ny,nf,nth)
        Ef = E.sum(axis=2) * dth  # (ny,nf)
        m0 = Ef.sum(axis=1) * DF
        m1 = (Ef * freq[None, :]).sum(axis=1) * DF
        S = (E * sth[None, None, :]).sum(axis=(1, 2))
        C = (E * cth[None, None, :]).sum(axis=(1, 2))
        Sxx = ((E * (n[:, :, None] * (1 + cth[None, None, :] ** 2) - 0.5)).sum(axis=(1, 2))) * dth * DF
        return E, Ef, m0, m1, S, C, Sxx

    def breaking(m0, m1, he, k, Ef):
        Hrms = np.sqrt(8 * np.maximum(m0, 0))
        fm = np.where(m0 > 0, m1 / np.maximum(m0, 1e-12), 0.1)
        if gam_k > 0:
            # Miche-type limit with mean wavenumber
            km = (Ef * k).sum(axis=1) / np.maximum(Ef.sum(axis=1), 1e-12)
            Hmax = gam_k / km * np.tanh(gam * km * he / gam_k)
        else:
            Hmax = gam * he + P.get("hbrk0", 0.0)
        b = Hrms / np.maximum(Hmax, 1e-6)
        Qb = qb_bj(b)
        D = 0.25 * alpha * fm * Qb * Hmax ** 2  # variance flux dissipation per unit x
        return Qb, D, Hrms, fm

    for i in range(nx):
        # record diagnostics at grid row i
        h, hx, hy = depth_at(x[i])
        wet = h > P.get("hwet", 0.0)
        he = np.maximum(h, hmin)
        k = wavenumber(sig[None, :], he[:, None])
        kh = k * he[:, None]
        n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
        cg = n * sig[None, :] / k
        E, Ef, m0, m1, S, C, Sxx = diagnostics(F, cg, k, he, n, i)
        Qb, D, Hrms, fm = breaking(m0, m1, he, k, Ef)
        Hs = 4 * np.sqrt(np.maximum(m0, 0))
        out["Hs"][i] = np.where(wet, Hs, np.nan)
        out["Tm"][i] = np.where(wet, m0 / np.maximum(m1, 1e-12), np.nan)
        out["Dm"][i] = np.where(wet, np.degrees(np.arctan2(S, C)), np.nan)
        out["Qb"][i] = np.where(wet, Qb, np.nan)
        out["Sxx"][i] = Sxx
        out["Dis"][i] = D
        out["Hrms"][i] = Hrms
        out["fm"][i] = fm
        if i == nx - 1:
            break
        for s in range(nsub):
            xx = x[i] + s * dx
            h, hx, hy = depth_at(xx + 0.5 * dx)
            he = np.maximum(h, hmin)
            k = wavenumber(sig[None, :], he[:, None])
            kh = k * he[:, None]
            n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
            cg = n * sig[None, :] / k
            # ---- lateral propagation (semi-Lagrangian, edge-clamped)
            shift = dx * tth / dy  # cells, per theta
            for j in range(nth):
                src = np.arange(ny) - shift[j]
                src = np.clip(src, 0, ny - 1)
                j0 = np.floor(src).astype(int)
                j0 = np.clip(j0, 0, ny - 2)
                w = src - j0
                F[:, :, j] = (1 - w)[:, None] * F[j0, :, j] + w[:, None] * F[j0 + 1, :, j]
            # ---- refraction in theta (conservative upwind FV with substeps)
            # c_theta = sig/sinh(2kh) * (sin th * hx - cos th * hy)
            fac = sig[None, :] / np.sinh(2 * np.minimum(kh, 30))  # (ny,nf)
            hxv = np.full(ny, hx) if np.ndim(hx) == 0 else hx
            cth_rate = fac[:, :, None] * (sth[None, None, :] * hxv[:, None, None] - cth[None, None, :] * hy[:, None, None])
            if current_u is not None:
                pass
            kap = cth_rate / (cg[:, :, None] * cth[None, None, :])  # dtheta/dx (ny,nf,nth)
            # face values
            kf = np.zeros((ny, nf, nth + 1))
            kf[:, :, 1:-1] = 0.5 * (kap[:, :, 1:] + kap[:, :, :-1])
            cfl = np.max(np.abs(kf)) * dx / dth
            nst = max(1, int(np.ceil(cfl / 0.8)))
            ddx = dx / nst
            for _ in range(nst):
                flux = np.where(kf[:, :, 1:-1] > 0, kf[:, :, 1:-1] * F[:, :, :-1], kf[:, :, 1:-1] * F[:, :, 1:])
                Fn = F.copy()
                Fn[:, :, :-1] -= ddx / dth * flux
                Fn[:, :, 1:] += ddx / dth * flux
                F = np.maximum(Fn, 0)
            # ---- breaking dissipation
            E = F / (cg[:, :, None] * cth[None, None, :])
            Ef = E.sum(axis=2) * dth
            m0 = Ef.sum(axis=1) * DF
            m1 = (Ef * freq[None, :]).sum(axis=1) * DF
            Qb, D, Hrms, fm = breaking(m0, m1, he, k, Ef)
            wgt = E * (freq[None, :, None] ** nexp)
            wsum = wgt.sum(axis=(1, 2)) * dth * DF
            # dF_i/dx = -D * wgt_i/wsum  -> exponential update per component
            rate = D[:, None, None] / np.maximum(wsum[:, None, None], 1e-12) * (freq[None, :, None] ** nexp) / (cg[:, :, None] * cth[None, None, :])
            F = F * np.exp(-rate * dx)
            # dry cells: kill energy
            F[h <= P.get("hwet", 0.0)] = 0.0
    return out


def qb_bj(b):
    """Solve (1-Q)/ln(Q) = -b^2 for Q (vectorised), root in (0, b^2)."""
    b = np.asarray(b, float)
    Q = np.zeros_like(b)
    m = b > 0.2
    bb = np.minimum(b[m], 1.0)
    lo = np.full_like(bb, -40.0)
    hi = np.log(np.maximum(bb * bb, 1e-300))
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        q = np.exp(mid)
        f = 1 - q + bb * bb * mid
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    q = np.exp(0.5 * (lo + hi))
    q[bb >= 1.0] = 1.0
    Q[m] = q
    return Q
