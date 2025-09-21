import timeit
import numpy as np

from hdvec.core import bind, similarity
from hdvec.encoding.fpe import generate_base, encode_fpe


def main() -> None:
    D = 65536
    base = generate_base(D)
    z1 = encode_fpe(0.7, base)
    z2 = encode_fpe(1.3, base)
    t_bind = timeit.timeit(lambda: bind(z1, z2), number=100)
    t_sim = timeit.timeit(lambda: similarity(z1, z2), number=100)
    print(f"D={D} bind: {t_bind:.4f}s/100, sim: {t_sim:.4f}s/100")


if __name__ == "__main__":
    main()
