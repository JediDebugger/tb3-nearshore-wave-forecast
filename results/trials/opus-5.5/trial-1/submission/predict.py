"""Nearshore wave forecast.  Usage: python predict.py conditions.json out_dir

Physics model calibrated on the 24 observed hours (see params.json):
  * bathymetry: Dean-type profile of the chart + bar re-located from the observations
    + rip channel through the bar (alongshore Gaussian) + coastal structure (land);
  * forward-marching half-plane spectral model (12 freq x 36 dir) with refraction,
    shoaling and Battjes-Janssen depth-limited breaking (wavemodel.py);
  * wave setup from the cross-shore radiation-stress gradient (one setup/wave iteration);
  * crest height Cs = Hs * min(cap, c0 + a * Ursell^b);
  * rip current u = -U0 G(y) X(x), U0 driven by Hs/h over the bar.
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wavemodel as wm  # noqa: E402

KEYS = ["Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u"]


def load_static():
    g = np.load(os.path.join(HERE, "data", "grid.npz"))
    P = json.load(open(os.path.join(HERE, "params.json")))
    return g["x"], g["y"], g["freq"], P


def bathymetry(x, y, P):
    X, Y = np.meshgrid(x, y, indexing="ij")
    A, xs = P["A"], P["xs"]
    h = A * np.maximum(xs - X, 0) ** (2.0 / 3.0) - P["beach"] * np.maximum(X - xs, 0)
    bar = P["bar_a"] * np.exp(-0.5 * ((X - P["bar_x"]) / P["bar_w"]) ** 2)
    G = np.exp(-0.5 * ((Y - P["ch_y"]) / P["ch_sig"]) ** 2)
    # channel: bar removed inside the channel, plus a deepening from the bar to the shore
    xt = 0.5 * (1 + np.tanh((X - P["ch_x0"]) / 20.0))
    h = h - bar * (1 - G) + P["ch_dep"] * G * xt * (h > 0)
    struct = (X >= P["struct_x"]) & (Y >= P["struct_y"])
    return h, struct


def setup(Sxx, depth, mwl0, tide, struct, fac):
    """Integrate d(eta)/dx = -fac/(h+eta) dSxx/dx column by column."""
    nx, ny = depth.shape
    eta = np.zeros((nx, ny))
    for i in range(1, nx):
        hh = depth[i - 1] + tide + eta[i - 1]
        hi = depth[i] + tide + eta[i - 1]
        ok = (hh > 0.05) & (hi > 0.0) & np.isfinite(Sxx[i]) & np.isfinite(Sxx[i - 1]) & ~struct[i]
        d = np.where(ok, -fac * (Sxx[i] - Sxx[i - 1]) / np.maximum(hh, 0.3), 0.0)
        d = np.clip(d, -0.03, 0.06)
        eta[i] = eta[i - 1] + d
    return tide + eta


def rip_current(x, y, depth, mwl, Hs, P):
    X, Y = np.meshgrid(x, y, indexing="ij")
    # forcing: Hs/h over the bar away from the channel
    jb = np.abs(y - P["ch_y"]) > 3 * P["ch_sig"]
    ib = (x >= P["bar_x"] - 30) & (x <= P["bar_x"] + 30)
    hb = (depth + mwl)[np.ix_(ib, jb)]
    r = np.nanmax(np.nanmean(Hs[np.ix_(ib, jb)] / np.maximum(hb, 0.05), axis=1))
    a1, r1 = P["rip_lin"]
    a2, r2 = P["rip_sqrt"]
    U0 = 0.5 * (a1 * max(0.0, r - r1) + a2 * np.sqrt(max(0.0, r - r2)))
    G = np.exp(-0.5 * ((Y - P["rip_y"]) / P["rip_sig"]) ** 2)
    Xs = 0.5 * (1 + np.tanh((X - P["rip_x0"]) / P["rip_L"]))
    htot = depth + mwl
    taper = np.clip(htot / 0.3, 0, 1)
    return -U0 * G * Xs * taper


def crest(Hs, Tm, htot, P):
    he = np.maximum(htot, 0.05)
    sig = 2 * np.pi / np.where(np.isfinite(Tm), Tm, 8.0)
    k = wm.wavenumber(sig, he)
    L = 2 * np.pi / k
    U = Hs * L ** 2 / he ** 3
    r = np.minimum(P["cs_cap"], P["cs_c0"] + P["cs_a"] * np.power(np.maximum(U, 0), P["cs_b"]))
    return Hs * r


def reflection(o, x, y, P, land):
    """Add the wave field reflected offshore by the structure face (energy R2 * incident)."""
    R2 = P.get("refl_R2", 0.0)
    Hs, Tm, Dm = o["Hs"].copy(), o["Tm"].copy(), o["Dm"].copy()
    if R2 <= 0:
        return Hs, Tm, Dm
    jf = np.where(y >= P["struct_y"])[0]
    i_face = np.max(np.where(x < P["struct_x"])[0])
    Ef = (Hs[i_face, jf] / 4.0) ** 2
    Tf = Tm[i_face, jf]
    thf = np.radians(Dm[i_face, jf])
    good = np.isfinite(Ef) & np.isfinite(Tf) & np.isfinite(thf)
    if not good.any():
        return Hs, Tm, Dm
    Ef, Tf, thf, yf = Ef[good], Tf[good], thf[good], y[jf][good]
    dy = y[1] - y[0]
    for i in range(i_face + 1):
        dist = x[i_face] - x[i]
        er = np.zeros(len(y)); sr = np.zeros(len(y)); cr = np.zeros(len(y)); m1r = np.zeros(len(y))
        yr = yf + dist * np.tan(thf)  # reflected ray keeps the alongshore component
        jj = np.round((yr - y[0]) / dy).astype(int)
        ok = (jj >= 0) & (jj < len(y))
        for a, b in zip(np.where(ok)[0], jj[ok]):
            e = R2 * Ef[a]
            er[b] += e; sr[b] += e * np.sin(thf[a]); cr[b] -= e * np.cos(thf[a]); m1r[b] += e / Tf[a]
        ei = (Hs[i] / 4.0) ** 2
        th = np.radians(Dm[i])
        okc = np.isfinite(ei) & (er > 0)
        et = ei + er
        Hs[i, okc] = 4 * np.sqrt(et[okc])
        Dm[i, okc] = np.degrees(np.arctan2(ei * np.sin(th) + sr, ei * np.cos(th) + cr))[okc]
        Tm[i, okc] = (et / (ei / Tm[i] + m1r))[okc]
    return Hs, Tm, Dm


def predict_case(c, x, y, freq, P):
    depth, struct = bathymetry(x, y, P)
    tide = float(c["tide"])
    land = struct.copy()
    dm = np.where(land, -5.0, depth)
    mwl = np.full(depth.shape, tide)
    for it in range(2):
        o = wm.run(c, dm, x, y, freq, P, mwl=mwl, land=land)
        mwl = setup(o["Sxx"], dm, mwl, tide, land, P["setup_fac"])
    o = wm.run(c, dm, x, y, freq, P, mwl=mwl, land=land)
    htot = dm + mwl
    wet = (htot > P.get("hwet_out", P["hwet"])) & ~land
    out = {}
    Hs_inc = o["Hs"]
    out["Hs"], out["Tm"], out["Dm"] = reflection(o, x, y, P, land)
    out["Dm"] = np.clip(out["Dm"], -89.0, 89.0)
    out["Qb"] = o["Qb"]
    out["mwl"] = mwl
    out["Cs"] = crest(o["Hs"], o["Tm"], htot, P)
    out["u"] = rip_current(x, y, dm, mwl, Hs_inc, P)
    # fill any wet cell left without a value from its offshore neighbour
    for k in KEYS:
        a = out[k].astype(float)
        for i in range(1, a.shape[0]):
            bad = wet[i] & ~np.isfinite(a[i])
            a[i, bad] = a[i - 1, bad]
        a[~wet] = np.nan
        out[k] = a.astype(np.float32)
    return out


def main(cond_file, out_dir):
    x, y, freq, P = load_static()
    os.makedirs(out_dir, exist_ok=True)
    for c in json.load(open(cond_file)):
        out = predict_case(c, x, y, freq, P)
        np.savez(os.path.join(out_dir, f"case_{c['case']}.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
