import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import angled_stratified
from diffaaable import aaa
from examples import material
from propagation import angled_sqrt
from scipy.constants import c as c0

if __name__ == "__main__":

    pol = "s"
    pol_idx = 1 if pol=="p" else 0

    TILDE = True

    kx = 3
    if TILDE:
        ext = 0.2
        k_r = jnp.concat([
            kx-jnp.logspace(ext, -3, 200), kx+jnp.logspace(-3, ext, 200)
        ])

        k_i = jnp.concat([
            -jnp.logspace(ext, -3, 200), jnp.logspace(-3, ext, 200)
        ])

    else:
        k_domain=jnp.array([-6-3j, 6+3j])
        k_r = jnp.linspace(k_domain[0].real, k_domain[1].real, 300)
        k_i = jnp.linspace(k_domain[0].imag, k_domain[1].imag, 301)

    K_r, K_i = jnp.meshgrid(k_r, k_i)

    #t_layer = jnp.linspace(0.02, 0.3, 9)
    t_layer = jnp.linspace(0.264, 0.272, 12)

    kx = jnp.ones_like(t_layer)*kx
    k0_mesh = K_r+ 1j*K_i
    downsample = 20

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]
    t_layer = t_layer[:, None, None]

    ns = [4]
    ds = [t_layer]

    num = len(t_layer)
    fig, axs = plt.subplots(3, int((num-0.1)//3+1), sharex=True, sharey=True)

    bc = 0*jnp.pi
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
    
    for bc in [-jnp.pi*3/2, jnp.pi/2]:# [1:] -3/2*jnp.pi, -jnp.pi, 0, jnp.pi, 3/2*jnp.pi
        smat_mesh = stack(
            bc_angle=bc, bc_arcsin_i=bc, bc_arcsin_j=bc
        )

        for idx, ax in enumerate(axs.flatten()):
            plt.sca(ax)
            K_tilde = angled_sqrt(
                (k0_mesh**2 - kx**2)[idx], 
                bc_angle=bc, 
                nan_tolerance=0)
        
            trans = smat_mesh[('in', 'out')][idx]

            if TILDE:
                kt = K_tilde[::downsample, ::downsample].flatten()
                tr =   trans[::downsample, ::downsample].flatten()
                filt = ~jnp.isnan(tr)
                kt = kt[filt]
                tr = tr[filt]
                z_j, f_j, w_j, z_n = aaa(
                    kt, 
                    tr, 
                    tol=1e-7
                )
            else:
                z_j, f_j, w_j, z_n = aaa(
                    k0_mesh[0, ::downsample, ::downsample], 
                    trans[::downsample, ::downsample], 
                    tol=1e-7
                )

            if TILDE:
                plt.pcolormesh(
                        K_tilde.real, K_tilde.imag, 
                        jnp.abs(trans), 
                        norm="log", 
                        vmax=1e2, 
                        vmin=1e-3
                )
            else: 
                plt.pcolormesh(K_r, K_i, trans, norm="log", vmin=1e-3, vmax=3)
            
            plt.scatter(z_n.real, z_n.imag, zorder=5)

    for ax in axs.flatten():
        ext_x = 4.5
        ext_y = 3.5
        ax.set_xlim([-ext_x, ext_x])
        ax.set_ylim([-ext_y, ext_y])
        ax.set_aspect(True)

    if TILDE:
        axs[1 ,0].set_ylabel(r"$\Im\{\tilde{k}\}$")
        axs[-1,1].set_xlabel(r"$\Re\{\tilde{k}\}$")
    else:
        axs[1 ,0].set_ylabel(r"$\Im\{k\}$")
        axs[-1,1].set_xlabel(r"$\Re\{k\}$")
    plt.show()