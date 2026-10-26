"""Stub: replace with your model. Usage: python predict.py conditions.json out_dir"""
import sys, os, json
import numpy as np

def main(cond_file, out_dir):
    g = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid.npz"))
    shape = (len(g["x"]), len(g["y"]))
    os.makedirs(out_dir, exist_ok=True)
    for c in json.load(open(cond_file)):
        out = {k: np.full(shape, np.nan, np.float32) for k in ["Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u"]}
        np.savez(os.path.join(out_dir, f"case_{c['case']}.npz"), **out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
