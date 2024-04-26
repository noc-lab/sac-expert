"""Helper functions used for policy updates."""
import numpy as np

def cg(f_Ax, b, cg_iters=20, residual_tol=1e-10):
    """Conjugate gradient sub-routine sourced from OpenAI Baselines."""
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for _ in range(cg_iters):
        z = f_Ax(p).numpy()
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x.astype('float32')