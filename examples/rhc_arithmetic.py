import numpy as np

from hdvec.residue import ResidueEncoder, res_add, crt_reconstruct


def main() -> None:
    D = 256
    moduli = [3, 5, 7]
    enc = ResidueEncoder(moduli=moduli, D=D)

    # Encode two integers and add in residue space
    x, y = 11, 14
    vx = enc(x)
    vy = enc(y)
    vsum = res_add(vx, vy)

    # CRT reconstruction demo (using direct residues for now)
    residues = np.array([x % m for m in moduli])
    recon = crt_reconstruct(residues, moduli)
    print("CRT recon of", residues, "=", recon)


if __name__ == "__main__":
    main()
