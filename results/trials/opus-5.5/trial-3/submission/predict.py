"""Nearshore wave forecast for the radar/video site.

Usage: python predict.py conditions.json out_dir

Stationary spectral (frequency-direction) wave model on the 10 m product grid:
  * x-marching energy-flux balance with implicit upwind alongshore transport,
    depth- and current-induced refraction, shoaling, wave-current interaction
  * Battjes-Janssen (1978) depth-limited breaking (gives Qb), JONSWAP bottom friction
  * partial reflection off the coastal structure (reflected field marched offshore)
  * 1D cross-shore wave setup from radiation-stress gradients
  * rip-current jet over the rip channel (active when waves break on the bar)
  * crest height from an Ursell-number-dependent crest/height ratio
Bathymetry, breaker parameters, reflection coefficient, rip and crest-height
relations were calibrated against the 24 hourly level-2 products in data/obs.
"""
import sys
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
DX = 10.0
DY = 10.0

# ---------------------------------------------------------------- calibration
# cross-shore depth profile below MSL (nodes in m from the buoy line); offshore
# of the first node the chart is used
PROFILE_NODES = [280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
PROFILE_DEPTH = [5.53, 5.39, 5.20, 4.68, 3.60, 2.54, 2.34, 3.07, 3.13, 2.86, 2.60, 2.44, 2.18, 1.95, 1.72, 1.45, 1.13, 0.84, 0.49, 0.00, -0.45, -0.90, -1.40, -1.90]
# rip channel: depth increase A * exp(-((y-yc)/wy)^2) * window(x1 < x < x2)
RIP_CHANNEL = dict(A=1.0, yc=380.0, wy=40.0, x1=370.0, x2=510.0)
GAMMA = 0.668
ALPHA = 0.769
CF_JON = 0.038
KR = 0.52                    # amplitude reflection coefficient of the structure face
THETA_MAX = 72.5            # half-width (deg) of the directional sector carried
CS_PAR = (0.5077, 0.006, 0.965, 0.80)   # Cs/Hs = min(c3, c0 + c1 * Ur^c2)
# rip jet: u = -U * exp(-((y-yc)/w)^2) * S(x);  U = RIP_K * Hs_bar * ramp(b_bar)
RIP_JET = dict(yc=380.9, w=51.0, x1=410.0, lx_off=123.0, x2=520.0, lx_on=30.0)
RIP_K = 0.145
RIP_B0, RIP_B1 = 0.533, 0.542
DRY_DEPTH = 0.0


# ---------------------------------------------------------------- utilities
def wavenumber(sig, d):
    d = np.maximum(d, 1e-3)
    x = sig * sig * d / G
    y = sig * np.sqrt(d / G)
    kh = x / np.power(1.0 - np.exp(-np.power(y, 2.5)), 0.4)
    for _ in range(2):
        t = np.tanh(kh)
        f = kh * t - x
        df = t + kh * (1 - t * t)
        kh = kh - f / df
    return kh / d


def qb_bj(b):
    """Battjes-Janssen fraction of breaking waves for b = Hrms/Hm."""
    b = np.asarray(b, float)
    q = np.zeros_like(b)
    m = b > 0.2
    bb = np.minimum(b[m], 1.0)
    Q = np.where(bb > 0.9, 0.9, np.exp(-1.0 / np.maximum(bb * bb, 1e-6)))
    Q = np.clip(Q, 1e-12, 1.0)
    for _ in range(30):
        lnQ = np.log(Q)
        f = Q - 1.0 - bb * bb * lnQ
        df = 1.0 - bb * bb / Q
        Q = np.clip(Q - f / np.where(np.abs(df) < 1e-12, -1e-12, df), 1e-12, 1.0)
    Q[bb >= 1.0] = 1.0
    q[m] = Q
    return q


def boundary_spectrum(cond, freq, th, tmax_deg):
    thf = np.radians(np.arange(-180, 180, 0.5) + 0.25)
    Ef = np.asarray(cond["Ef"], float)
    out = np.zeros((len(freq), len(th)))
    for i in range(len(freq)):
        s = float(cond["sf"][i])
        dm = np.radians(float(cond["dirf"][i]))
        Dn = np.abs(np.cos((thf - dm) / 2.0)) ** (2.0 * s)
        norm = Dn.sum() * np.radians(0.5)
        Dj = np.abs(np.cos((th - dm) / 2.0)) ** (2.0 * s) / max(norm, 1e-12)
        out[i] = max(Ef[i], 0.0) * Dj
    out[:, np.abs(th) > np.radians(tmax_deg) + 1e-9] = 0.0
    return out


def load_site():
    g = np.load(os.path.join(HERE, "data", "grid.npz"))
    ch = np.load(os.path.join(HERE, "data", "chart.npz"))
    x, y, freq = g["x"], g["y"], g["freq"]
    prof = np.interp(x, ch["x"], ch["depth_msl"].mean(axis=1))
    if PROFILE_NODES is not None:
        nodes = np.asarray(PROFILE_NODES, float)
        m = x >= nodes[0]
        prof[m] = np.interp(x[m], nodes, np.asarray(PROFILE_DEPTH, float))
    XX, YY = np.meshgrid(x, y, indexing="ij")
    r = RIP_CHANNEL
    sx = 1 / (1 + np.exp(-(XX - r["x1"]) / 10.0)) / (1 + np.exp((XX - r["x2"]) / 10.0))
    h0 = prof[:, None] + r["A"] * np.exp(-((YY - r["yc"]) / r["wy"]) ** 2) * sx
    # structure mask on the product grid (chart cells -> 10 m cells)
    st = np.zeros(XX.shape, bool)
    ys = ch["y"][ch["structure"].any(axis=0)]
    xs = ch["x"][ch["structure"].any(axis=1)]
    if len(ys) and len(xs):
        st = (XX >= xs.min()) & (YY >= ys.min())
    return x, y, freq, h0, st, prof


def rip_template(x, y):
    j = RIP_JET
    XX, YY = np.meshgrid(x, y, indexing="ij")
    gy = np.exp(-((YY - j["yc"]) / j["w"]) ** 2)
    sx = np.where(XX < j["x1"], np.exp(-((XX - j["x1"]) / j["lx_off"]) ** 2),
                  np.where(XX > j["x2"], np.exp(-((XX - j["x2"]) / j["lx_on"]) ** 2), 1.0))
    return gy * sx


# ---------------------------------------------------------------- wave model
class WaveModel:
    def __init__(self, x, y, freq, h0, struct):
        self.x, self.y, self.freq = x, y, freq
        self.nx, self.ny, self.nf = len(x), len(y), len(freq)
        self.h0, self.struct = h0, struct
        dth = 5.0
        n = int(round(2 * THETA_MAX / dth))
        self.th = np.radians(np.linspace(-THETA_MAX + dth / 2, THETA_MAX - dth / 2, n))
        self.dth = np.radians(dth)
        self.df = float(np.median(np.diff(freq))) if len(freq) > 1 else 0.02
        self.sig = 2 * np.pi * freq

    def kin(self, d):
        dd = np.maximum(d, 0.01)[None]
        s = self.sig[:, None, None]
        k = wavenumber(s, dd)
        kh = np.minimum(k * dd, 30)
        c = s / k
        n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
        return k, c, n * c, n

    def march(self, F0, d, wet, c, cg, k, u, src=None, E_other=None):
        """March energy flux F (nf, nth, ny) from row 0 to the last row.
        u: cross-shore current in the marching frame (nx, ny)."""
        nf, nth, ny, nx = self.nf, len(self.th), self.ny, d.shape[0]
        th = self.th
        cos, sin = np.cos(th), np.sin(th)
        E = np.zeros((nf, nth, nx, ny))
        cxg = np.gradient(c, DX, axis=1)
        cyg = np.gradient(c, DY, axis=2)
        uxg = np.gradient(u, DX, axis=0)
        uyg = np.gradient(u, DY, axis=1)
        sig = self.sig
        pos = sin > 0
        neg = sin < 0
        F = None
        for i in range(nx):
            cgi = cg[:, i, :][:, None, :]
            cxi = cgi * cos[None, :, None] + u[i][None, None, :]
            cxi = np.maximum(cxi, 0.3 * cgi * cos[None, :, None])
            if F is None:
                F = F0.copy()
                if src is not None and i in src:
                    F = F + src[i]
            else:
                if src is not None and i in src:
                    F = F + src[i]
                # alongshore transport, implicit upwind, velocity ratio a = cy/cx
                a = cgi * sin[None, :, None] / cxi
                mu = np.abs(a) * DX / DY
                Fp = F[:, pos, :]
                mp = mu[:, pos, :]
                for j in range(1, ny):
                    Fp[:, :, j] = (Fp[:, :, j] + mp[:, :, j - 1] * Fp[:, :, j - 1]) / (1 + mp[:, :, j])
                F[:, pos, :] = Fp
                Fm = F[:, neg, :]
                mm = mu[:, neg, :]
                for j in range(ny - 2, -1, -1):
                    Fm[:, :, j] = (Fm[:, :, j] + mm[:, :, j + 1] * Fm[:, :, j + 1]) / (1 + mm[:, :, j])
                F[:, neg, :] = Fm
                # refraction (depth + current), explicit upwind with sub-steps
                ci = c[:, i, :][:, None, :]
                cth = cgi * (sin[None, :, None] * cxg[:, i, :][:, None, :]
                             - cos[None, :, None] * cyg[:, i, :][:, None, :]) / ci
                cth = cth - cos[None, :, None] * (-sin[None, :, None] * uxg[i][None, None, :]
                                                  + cos[None, :, None] * uyg[i][None, None, :])
                rate = np.where(wet[i][None, None, :], cth / cxi, 0.0)
                cfl = np.max(np.abs(rate)) * DX / self.dth
                if cfl > 0:
                    nsub = min(int(np.ceil(cfl / 0.8)), 60)
                    dt = DX / nsub
                    rf = 0.5 * (rate[:, 1:, :] + rate[:, :-1, :])
                    rf = np.clip(rf, -0.8 * self.dth / dt, 0.8 * self.dth / dt)
                    for _ in range(nsub):
                        up = np.where(rf > 0, F[:, :-1, :], F[:, 1:, :])
                        flux = rf * up * (dt / self.dth)
                        F[:, :-1, :] -= flux
                        F[:, 1:, :] += flux
                    F = np.maximum(F, 0.0)
            Ei = F / cxi
            Etot = Ei.sum(axis=(0, 1)) * self.df * self.dth
            Eall = Etot + (E_other[i] if E_other is not None else 0.0)
            di = np.maximum(d[i], 0.0)
            Hrms = np.sqrt(8 * np.maximum(Eall, 0))
            Hm = GAMMA * di
            q = qb_bj(np.where(Hm > 1e-6, Hrms / np.maximum(Hm, 1e-6), 10.0))
            m1 = (Ei.sum(axis=1) * (sig[:, None] / (2 * np.pi))).sum(0) * self.df * self.dth
            fm = np.where(Etot > 1e-12, m1 / np.maximum(Etot, 1e-12), 0.1)
            D = 0.25 * ALPHA * fm * q * Hm * Hm
            kap = np.where(Eall > 1e-12, D / np.maximum(Eall, 1e-12), 0.0)
            khi = np.minimum(k[:, i, :] * np.maximum(di, 0.01), 30)
            fr = CF_JON * (sig[:, None] ** 2) / (G ** 2 * np.sinh(khi) ** 2)
            F = F * np.exp(-DX * (kap[None, None, :] + fr[:, None, :]) / cxi)
            F = np.where(wet[i][None, None, :], F, 0.0)
            E[:, :, i, :] = F / cxi
        return E

    def solve(self, cond, u=None, n_iter=2):
        tide = float(cond["tide"])
        nx, ny = self.nx, self.ny
        th = self.th
        if u is None:
            u = np.zeros((nx, ny))
        eta = np.full((nx, ny), tide)
        Eb = boundary_spectrum(cond, self.freq, th, THETA_MAX)
        for _ in range(n_iter):
            d = self.h0 + eta
            wet = (d > DRY_DEPTH) & (~self.struct)
            k, c, cg, n = self.kin(d)
            F0 = (Eb * (cg[:, 0, 0][:, None] * np.cos(th)[None, :] + u[0, 0]))[:, :, None] * np.ones(ny)[None, None, :]
            Ein = self.march(F0, d, wet, c, cg, k, u)
            Eref = None
            if KR > 0:
                src = {}
                for j in range(ny):
                    col = self.struct[:, j]
                    if col.any():
                        i0 = int(np.argmax(col))
                        if i0 > 0 and wet[i0 - 1, j]:
                            Fi = Ein[:, :, i0 - 1, j] * cg[:, i0 - 1, j][:, None] * np.cos(th)[None, :]
                            src.setdefault(i0 - 1, np.zeros((self.nf, len(th), ny)))
                            src[i0 - 1][:, :, j] += KR ** 2 * Fi
                if src:
                    imax = max(src.keys())
                    sub = slice(0, imax + 1)
                    srcf = {imax - i: v for i, v in src.items()}
                    Etot_in = Ein.sum(axis=(0, 1)) * self.df * self.dth
                    Er = self.march(np.zeros((self.nf, len(th), ny)), d[sub][::-1], wet[sub][::-1],
                                    c[:, sub][:, ::-1], cg[:, sub][:, ::-1], k[:, sub][:, ::-1],
                                    -u[sub][::-1], src=srcf, E_other=Etot_in[sub][::-1])
                    Eref = np.zeros_like(Ein)
                    Eref[:, :, sub] = Er[:, :, ::-1]
            Etot = Ein + (Eref if Eref is not None else 0.0)
            cth2 = np.cos(th) ** 2
            Sxx = (Etot * (n[:, None] * (1 + cth2[None, :, None, None]) - 0.5)).sum(axis=(0, 1)) * self.df * self.dth
            eta_new = np.full((nx, ny), tide)
            for i in range(1, nx):
                dd = np.maximum(self.h0[i] + eta_new[i - 1], 0.05)
                deta = -(Sxx[i] - Sxx[i - 1]) / dd
                ok = wet[i] & wet[i - 1]
                eta_new[i] = np.where(ok, eta_new[i - 1] + deta, eta_new[i - 1])
            eta = eta_new
        d = self.h0 + eta
        wet = (d > DRY_DEPTH) & (~self.struct)
        return self.outputs(Ein, Eref, d, wet, eta, k)

    def outputs(self, Ein, Eref, d, wet, eta, k):
        th = self.th
        dA = self.df * self.dth
        fr = self.freq[:, None, None, None]
        c4 = np.cos(th)[None, :, None, None]
        s4 = np.sin(th)[None, :, None, None]
        m0 = Ein.sum(axis=(0, 1)) * dA
        m1 = (Ein * fr).sum(axis=(0, 1)) * dA
        sx = (Ein * c4).sum(axis=(0, 1))
        sy = (Ein * s4).sum(axis=(0, 1))
        if Eref is not None:
            m0 = m0 + Eref.sum(axis=(0, 1)) * dA
            m1 = m1 + (Eref * fr).sum(axis=(0, 1)) * dA
            sx = sx - (Eref * c4).sum(axis=(0, 1))
            sy = sy + (Eref * s4).sum(axis=(0, 1))
        Hs = 4 * np.sqrt(np.maximum(m0, 0))
        Tm = np.where(m1 > 1e-12, m0 / np.maximum(m1, 1e-12), 1.0 / self.freq.mean())
        Dm = np.degrees(np.arctan2(sy, sx))
        di = np.maximum(d, 0)
        Hm = GAMMA * di
        Qb = qb_bj(np.where(Hm > 1e-6, Hs / np.sqrt(2) / np.maximum(Hm, 1e-6), 10.0))
        # crest height
        dd = np.maximum(d, 0.05)
        kk = wavenumber(2 * np.pi / np.maximum(Tm, 1.0), dd)
        L = 2 * np.pi / kk
        Ur = Hs * L ** 2 / dd ** 3
        c0, c1, c2, c3 = CS_PAR
        Cs = Hs * np.minimum(c3, c0 + c1 * np.power(np.maximum(Ur, 0), c2))
        out = dict(Hs=Hs, Tm=Tm, Dm=Dm, Qb=Qb, mwl=eta.copy(), Cs=Cs)
        for kk_ in out:
            out[kk_] = np.where(wet, out[kk_], np.nan)
        out["_d"] = d
        out["_wet"] = wet
        return out


def rip_amplitude(out, x, y, prof_bar_rows):
    """rip jet speed from breaking intensity on the bar (south of the channel)."""
    Hs = out["Hs"]
    d = out["_d"]
    js = slice(0, 26)
    best = 0.0
    hsb = 0.0
    for i in prof_bar_rows:
        h = np.nanmean(Hs[i, js])
        dd = np.nanmean(d[i, js])
        if not np.isfinite(h) or not np.isfinite(dd) or dd <= 0.05:
            continue
        b = h / np.sqrt(2) / (GAMMA * dd)
        if b > best:
            best, hsb = b, h
    ramp = np.clip((best - RIP_B0) / (RIP_B1 - RIP_B0), 0.0, 1.0)
    return RIP_K * hsb * ramp, best


def predict_case(cond, site):
    x, y, freq, h0, st, prof = site
    model = WaveModel(x, y, freq, h0, st)
    out = model.solve(cond)
    # bar crest rows (shallowest point of the profile between 330 and 470 m)
    m = (x >= 330) & (x <= 470)
    ib = np.where(m)[0][np.argmin(prof[m])]
    bar_rows = range(max(ib - 2, 0), min(ib + 3, len(x)))
    U, b = rip_amplitude(out, x, y, bar_rows)
    T = rip_template(x, y)
    u = -U * T
    if U > 1e-3:
        out = model.solve(cond, u=u * out["_wet"])
        U2, _ = rip_amplitude(out, x, y, bar_rows)
        u = -U2 * T
    res = {k: out[k].astype(np.float32) for k in ["Hs", "Tm", "Dm", "Qb", "mwl", "Cs"]}
    res["u"] = np.where(out["_wet"], u, np.nan).astype(np.float32)
    return res


def main(cond_file, out_dir):
    site = load_site()
    os.makedirs(out_dir, exist_ok=True)
    conds = json.load(open(cond_file))
    if isinstance(conds, dict):
        conds = [conds]
    for i, c in enumerate(conds):
        case = c.get("case", i)
        res = predict_case(c, site)
        np.savez(os.path.join(out_dir, f"case_{case}.npz"), **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
