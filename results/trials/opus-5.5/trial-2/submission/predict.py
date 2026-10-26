"""Nearshore wave forecast. Usage: python predict.py conditions.json out_dir

Phase-averaged spectral wave model on the 10 m product grid:
  - boundary: buoy spectrum (cos-2s spreading per frequency bin) at x = 0
  - onshore march (semi-Lagrangian in y and theta): shoaling, depth refraction,
    Battjes-Janssen depth-limited breaking (gamma, alpha calibrated on the record)
  - partial reflection from the seaward face of the coastal structure, marched offshore
  - wave set-up from the cross-shore radiation-stress balance (iterated with the waves)
  - crest height Cs = Hs * min(0.8, 0.5 + a Ur^b), calibrated on the stereo-video record
  - cross-shore surface current: rip jet in the bar channel, active at low tide
Bathymetry: chart profile corrected with the bar position/depth and the rip channel
inferred from the radar/video record (stored in model_params.npz next to this file).
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
g = 9.81


def load_static():
    G = np.load(os.path.join(HERE, "data", "grid.npz"))
    P = dict(np.load(os.path.join(HERE, "model_params.npz")))
    return G, P


def wavenumber(sig, h):
    h = np.maximum(h, 1e-3)
    k0 = sig ** 2 / g
    k = k0 / np.tanh(np.maximum((k0 * h) ** 0.75, 1e-9)) ** (2 / 3.)
    for _ in range(4):
        t = np.tanh(k * h)
        k = k - (g * k * t - sig ** 2) / (g * t + g * k * h * (1 - t ** 2))
    return k


def qb_bj(b):
    b = np.asarray(b, float)
    Q = np.zeros_like(b)
    m = b > 0.2
    if m.any():
        bb = np.minimum(b[m], 1.0)
        q = np.clip(np.exp(-1 / bb ** 2), 1e-12, 1)
        for _ in range(40):
            lq = np.log(np.maximum(q, 1e-300))
            f = (1 - q) + bb ** 2 * lq
            df = -1 + bb ** 2 / np.maximum(q, 1e-300)
            q = np.clip(q - f / df, 1e-12, 1.0)
        q = np.where(bb >= 0.9999, 1.0, q)
        Q[m] = q
    return Q


def boundary_spectrum(c, freq, th):
    E = np.maximum(np.asarray(c["Ef"], float), 0.0)
    d = np.radians(np.asarray(c["dirf"], float))
    s = np.maximum(np.asarray(c["sf"], float), 0.1)
    thf = np.linspace(-np.pi, np.pi, 1441)[:-1]
    ddf = (thf[None, :] - d[:, None] + np.pi) % (2 * np.pi) - np.pi
    norm = (np.cos(ddf / 2) ** (2 * s[:, None])).sum(1) * (thf[1] - thf[0])
    dd = (th[None, :] - d[:, None] + np.pi) % (2 * np.pi) - np.pi
    return E[:, None] * np.cos(dd / 2) ** (2 * s[:, None]) / norm[:, None]


def shift_y(F, s):
    nf, nth, ny = F.shape
    out = np.empty_like(F)
    idx = np.arange(ny)
    for j in range(nth):
        src = np.clip(idx - s[j], 0, ny - 1)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, ny - 1)
        w = src - i0
        out[:, j, :] = F[:, j, i0] * (1 - w) + F[:, j, i1] * w
    return out


def shift_th(F, s):
    nf, nth, ny = F.shape
    out = np.zeros_like(F)
    tgt = np.arange(nth)[None, :, None] + s
    i0 = np.floor(tgt).astype(int)
    w = tgt - i0
    fi = np.broadcast_to(np.arange(nf)[:, None, None], F.shape)
    yi = np.broadcast_to(np.arange(ny)[None, None, :], F.shape)
    for ii, ww in ((i0, 1 - w), (i0 + 1, w)):
        ok = (ii >= 0) & (ii < nth)
        np.add.at(out, (fi[ok], ii[ok], yi[ok]), (F * ww)[ok])
    return out


def march(E0, hh, wet, sig, freq, th, dx, gamma, alpha, df, i_start=0, inject=None, brk=True):
    """March energy density E(f,th,y) through rows i_start..nx-1 of hh (waves travel toward +row).
    inject(i, E) -> E lets the caller add energy at a row. Returns per-row spectra list."""
    nx, ny = hh.shape
    dth = th[1] - th[0]
    k = wavenumber(sig[:, None, None], hh[None])
    cg = 0.5 * (1 + 2 * k * hh / np.sinh(np.minimum(2 * k * hh, 50))) * sig[:, None, None] / k
    dhdy = np.gradient(hh, dx, axis=1)
    cos = np.cos(th)[None, :, None]
    sin = np.sin(th)[None, :, None]
    E = E0.copy()
    specs = [None] * nx
    Qbs = np.zeros((nx, ny))
    for i in range(i_start, nx):
        if inject is not None:
            E = inject(i, E)
        E[:, :, ~wet[i]] = 0.0
        specs[i] = E.copy()
        m0 = E.sum((0, 1)) * df * dth
        m1 = (E.sum(1) * freq[:, None]).sum(0) * df * dth
        Hrms = np.sqrt(8 * np.maximum(m0, 0))
        Hm = gamma * hh[i]
        Qb = qb_bj(Hrms / np.maximum(Hm, 1e-6))
        Qbs[i] = np.where(wet[i], Qb, 0)
        if i == nx - 1:
            break
        cgi = cg[:, i, :][:, None, :]
        F = cgi * cos * E
        if brk:
            fm = m1 / np.maximum(m0, 1e-12)
            Dbr = 0.25 * alpha * fm * Qb * Hm ** 2
            Fx = F.sum((0, 1)) * df * dth
            F = F * np.exp(-Dbr * dx / np.maximum(Fx, 1e-9))[None, None, :]
        F = shift_y(F, np.tan(th))
        cgm = 0.5 * (cg[:, i, :] + cg[:, i + 1, :])
        km = 0.5 * (k[:, i, :] + k[:, i + 1, :])
        hm_ = 0.5 * (hh[i] + hh[i + 1])
        dhx = (hh[i + 1] - hh[i]) / dx
        dhy = 0.5 * (dhdy[i] + dhdy[i + 1])
        sk = sig[:, None] / np.sinh(np.minimum(2 * km * hm_[None, :], 50))
        cth = sk[:, None, :] * (sin * dhx - cos * dhy)
        F = shift_th(F, cth / (cgm[:, None, :] * cos) * dx / dth)
        E = F / (cg[:, i + 1, :][:, None, :] * cos)
    return specs, Qbs, k, cg


def solve_case(c, G, P):
    x = G["x"]; y = G["y"]; freq = G["freq"]; df = float(P["df"])
    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0])
    sig = 2 * np.pi * freq
    th = np.radians(np.linspace(-87.5, 87.5, 36))
    dth = th[1] - th[0]
    depth = P["depth"]
    struct = P["struct"].astype(bool)
    gamma = float(P["gamma"]); alpha = float(P["alpha"])
    tide = float(c["tide"])
    E0 = boundary_spectrum(c, freq, th)
    E0 = np.repeat(E0[:, :, None], ny, axis=2)
    eta = np.zeros((nx, ny))
    hmin = float(P["hmin"])
    for it in range(3):
        mwl = tide + eta
        h = depth + mwl
        wet = (h > hmin) & ~struct
        hh = np.maximum(h, 0.05)
        specs, Qb, k, cg = march(E0, hh, wet, sig, freq, th, dx, gamma, alpha, df)
        Ei = np.stack(specs, 0)  # nx,nf,nth,ny
        # reflection from structure face
        Er = np.zeros_like(Ei)
        R2 = float(P["R2"])
        if R2 > 0 and struct.any():
            face = np.zeros((nx, ny), bool)
            face[:-1] = struct[1:] & ~struct[:-1]
            rows = np.unique(np.where(face)[0])
            hf = hh[::-1].copy()
            wetf = wet[::-1].copy()
            def inject(ii, E):
                i = nx - 1 - ii
                if i in rows:
                    m = face[i]
                    E = E.copy()
                    E[:, :, m] += R2 * Ei[i][:, :, m] * (np.cos(th)[None, :, None] > 0)
                return E
            i0 = nx - 1 - rows.max()
            specs_r, _, _, _ = march(np.zeros_like(E0), hf, wetf, sig, freq, th, dx, gamma, alpha, df,
                                     i_start=i0, inject=inject, brk=False)
            for ii in range(i0, nx):
                Er[nx - 1 - ii] = specs_r[ii]
        cos = np.cos(th)[None, None, :, None]; sin = np.sin(th)[None, None, :, None]
        n = cg / (sig[:, None, None] / k)  # nf,nx,ny
        n = np.transpose(n, (1, 0, 2))[:, :, None, :]
        Etot = Ei + Er
        Sxx = ((n * (1 + cos ** 2) - 0.5) * Etot).sum((1, 2)) * df * dth
        # set-up: d eta/dx = -1/h dSxx/dx, per alongshore column
        dS = np.diff(Sxx, axis=0)
        new = np.zeros((nx, ny))
        for i in range(1, nx):
            hm = np.maximum(0.5 * (h[i] + h[i - 1]) + 0 * new[i - 1], 0.05)
            step = -dS[i - 1] / hm
            ok = wet[i] & wet[i - 1]
            new[i] = np.where(ok, new[i - 1] + step, new[i - 1])
        eta = 0.5 * eta + 0.5 * new if it > 0 else new
    m0i = Ei.sum((1, 2)) * df * dth
    m0r = Er.sum((1, 2)) * df * dth
    m0 = m0i + m0r
    m1 = ((Ei + Er).sum(2) * freq[None, :, None]).sum(1) * df * dth
    Hs = 4 * np.sqrt(np.maximum(m0, 0))
    Tm = m0 / np.maximum(m1, 1e-12)
    sx = (Ei * sin[0]).sum((1, 2)) + (Er * sin[0]).sum((1, 2))
    cx = (Ei * cos[0]).sum((1, 2)) - (Er * cos[0]).sum((1, 2))
    Dm = np.degrees(np.arctan2(sx, cx))
    mwl = tide + eta
    h = depth + mwl
    # crest height
    hl = np.maximum(h, 0.05)
    kk = wavenumber(2 * np.pi / np.maximum(Tm, 0.5), hl)
    Ur = Hs * (2 * np.pi / kk) ** 2 / hl ** 3
    Cs = Hs * np.minimum(float(P["cs_cap"]), 0.5 + float(P["cs_a"]) * Ur ** float(P["cs_b"]))
    # rip current
    Hbar = np.nanmean(Hs[int(P["bar_row"]), P["ref_cols"].astype(int)])
    S = 1.0 / (1.0 + np.exp((tide - float(P["rip_t0"])) / float(P["rip_tw"])))
    u = float(P["rip_k"]) * Hbar * S * P["rip_shape"]
    wet = (h > hmin) & ~struct
    # cells that only became wet with the final set-up: carry the swash values onshore
    Hs, Tm, Dm, Qb, Cs = (a.copy() for a in (Hs, Tm, Dm, Qb, Cs))
    for i in range(1, nx):
        m = wet[i] & (Hs[i] <= 1e-6) & wet[i - 1]
        if m.any():
            for a in (Hs, Tm, Dm, Qb, Cs):
                a[i, m] = a[i - 1, m]
    out = {"Hs": Hs, "Tm": Tm, "Dm": Dm, "Qb": Qb, "mwl": mwl, "Cs": Cs, "u": u}
    # fill any non-finite wet values from neighbours (safety)
    for kname, a in out.items():
        a = np.where(np.isfinite(a), a, np.nan)
        if np.isnan(a[wet]).any():
            from scipy.ndimage import generic_filter
            fill = generic_filter(a, np.nanmean, size=3, mode="nearest")
            a = np.where(np.isnan(a), fill, a)
            a = np.where(np.isnan(a), 0.0, a)
        out[kname] = np.where(wet, a, np.nan).astype(np.float32)
    return out


def main(cond_file, out_dir):
    G, P = load_static()
    os.makedirs(out_dir, exist_ok=True)
    conds = json.load(open(cond_file))
    if isinstance(conds, dict):
        conds = [conds]
    for c in conds:
        out = solve_case(c, G, P)
        np.savez(os.path.join(out_dir, f"case_{c['case']}.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
