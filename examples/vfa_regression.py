import numpy as np

from hdvec.fpe import FPEEncoder, generate_base, encode_fpe
from hdvec.vfa import VFAEncoder, encode_function, readout
from hdvec.core import bind


def main() -> None:
    D = 256
    base = generate_base(D)
    enc = FPEEncoder(D)

    # Encode a simple function y = sum alpha_k z(r_k)
    points = np.array([0.2, 0.8])
    alphas = np.array([1.0, -0.5])
    y_f = encode_function(points, alphas, base)

    val = readout(y_f, 0.5, base)
    print("readout@0.5:", val)

    # Shift by t=0.1 using bind with z(t)
    z_t = encode_fpe(0.1, base)
    y_shift = bind(y_f, z_t)
    print("shifted sim delta:", (y_shift.conj() * y_f).sum().real / D)


if __name__ == "__main__":
    main()
