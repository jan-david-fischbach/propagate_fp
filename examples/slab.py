import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import angled_stratified
from diffaaable import aaa
from examples import material
from propagation import angled_sqrt
from scipy.constants import c as c0
from numpy.linalg import LinAlgError

def tildify(k, Cs):
    return 1/len(Cs)*jnp.sum([
        angled_sqrt(
            (k**2 - C**2)[idx], 
            bc_angle=bc, 
            nan_tolerance=0) for C in Cs
    ])

if __name__ == "__main__":

    pol = "s"
    pol_idx = 1 if pol=="p" else 0
    kx = 3
    n_sub = 2
    n_sup = 1

    TILDE = True
    SAMPLE_REAL = True
    num_samples = 80

    ext = 0.6
    ixt = -2
    k_r = jnp.concat([
        kx-jnp.logspace(ext, ixt, 200), kx+jnp.logspace(ixt, ext, 200)
    ])

    k_r = jnp.sort(jnp.concat([k_r, -k_r]))

    k_i = jnp.concat([
        -jnp.logspace(ext, ixt, 200), jnp.logspace(ixt, ext, 200)
    ])

    if SAMPLE_REAL:
        downsample = len(k_r)//num_samples
    else:
        downsample = int(jnp.sqrt(len(k_r)*len(k_i)//num_samples))


    K_r, K_i = jnp.meshgrid(k_r, k_i)

    t_layer = jnp.linspace(0.02, 0.3, 9)
    #t_layer = jnp.linspace(0.264, 0.272, 12)

    kx = jnp.ones_like(t_layer)*kx
    k0_mesh = K_r+ 1j*K_i

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]
    t_layer = t_layer[:, None, None]

    ns = [n_sub, 4+1j, n_sup]
    ds = [t_layer]

    num = len(t_layer)
    fig, axs = plt.subplots(3, int((num-0.1)//3+1), sharex=True, sharey=True)

    bc = 0*jnp.pi
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
    
    for i, bc in enumerate([-jnp.pi*3/2, jnp.pi/2][not TILDE:]):# -3/2*jnp.pi, -jnp.pi, 0, jnp.pi, 3/2*jnp.pi
        smat_mesh = stack(
            bc_angle=bc, bc_arcsin_i=bc, bc_arcsin_j=bc
        )

        smat_real = stack(
            bc_angle=bc, bc_arcsin_i=bc, bc_arcsin_j=bc, k0=k_r
        )

        for idx, ax in enumerate(axs.flatten()):
            plt.sca(ax)
            
            trans      = smat_mesh[('in', 'out')][idx]
            trans_real = smat_real[('in', 'out')][idx][0]

            K_tilde = angled_sqrt((k0_mesh**2 - kx**2)[idx], 
                            bc_angle=bc, 
                            nan_tolerance=0)
            
            if TILDE:
                if SAMPLE_REAL:
                    k_r_tilde = angled_sqrt(
                        (k_r**2 - jnp.squeeze(kx[idx])**2), 
                        bc_angle=bc, 
                        nan_tolerance=0)
                    kt = k_r_tilde[::downsample]
                    tr = trans_real[::downsample]
                else:
                    kt = K_tilde[::downsample, ::downsample].flatten()
                    tr =   trans[::downsample, ::downsample].flatten()
            else:
                if SAMPLE_REAL:
                    kt = k_r[::downsample]
                    tr = trans_real[::downsample]
                else:
                    kt = k0_mesh[0, ::downsample, ::downsample].flatten()
                    tr =   trans[::downsample, ::downsample].flatten()
            
            filt = ~jnp.isnan(tr)
            kt = kt[filt]
            tr = tr[filt]
            z_j, f_j, w_j, z_n = aaa(
                kt, 
                tr, 
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
                plt.pcolormesh(K_r, K_i, jnp.abs(trans), norm="log", vmin=1e-3, vmax=3)
            
            if bc>0:
                plt.plot(k_r, jnp.abs(trans_real), "k")
            plt.grid()
            
            color=f"C{i}"
            plt.scatter(z_n.real, z_n.imag, zorder=5, marker="x", color=color)
            plt.scatter(kt.real, kt.imag, zorder=5, marker=".", color=color)

    for ax in axs.flatten():
        ext_x = 6
        ext_y = 5
        ax.set_xlim([-ext_x, ext_x])
        ax.set_ylim([-ext_y, ext_y])
        ax.set_aspect(True)

    if TILDE:
        axs[1 ,0].set_ylabel(r"$\Im\{\tilde{k}\}$")
        axs[-1,1].set_xlabel(r"$\Re\{\tilde{k}\}$")
    else:
        axs[1 ,0].set_ylabel(r"$\Im\{k\}$")
        axs[-1,1].set_xlabel(r"$\Re\{k\}$")
    plt.xticks([-3, 0, 3])
    plt.show()