import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import angled_stratified
from diffaaable import aaa
from examples import material
from propagation import angled_sqrt
from scipy.constants import c as c0

def k_to_wfreq(k0):
    return k0 * c0 / 300e12

def k_to_f(k0):
    return 1/(2*jnp.pi) * k0 * c0

def f_to_k(f):
    return 2*jnp.pi * f / c0

def pcolormesh_k(K_r, K_i, val, **kwargs):
    plt.pcolormesh(k_to_f(K_r)/1e12, k_to_f(K_i)/1e12, val, **kwargs)
    plt.xlabel("$\Re\{f\}$ [THz]")
    plt.ylabel("$\Im\{f\}$ [THz]")


if __name__ == "__main__":
    to_Hz = 300e12/(2*jnp.pi)
    #f_domain=jnp.array([7.5-30j, 10.5+0.2j])*to_Hz
    #f_domain=jnp.array([7.5-0.3j, 10.5+0.03j])*to_Hz
    f_domain=jnp.array([7.5-0.1j, 10.5+0.01j]) *to_Hz
    k_domain=f_to_k(f_domain)

    pol = "s"
    pol_idx = 1 if pol=="p" else 0

    k_r = jnp.linspace(k_domain[0].real, k_domain[1].real, 300)
    k_i = jnp.linspace(k_domain[0].imag, k_domain[1].imag, 301)
    K_r, K_i = jnp.meshgrid(k_r, k_i)

    t_layer = jnp.ones(9)*0.3e-6 # jnp.linspace(0.01, 0.09, 9)*1e-6
    kx = jnp.linspace(0, 2, 9) * max(k_domain.real)
    k0_mesh = K_r+ 1j*K_i

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]
    t_layer = t_layer[:, None, None]

    wfreq = k_to_wfreq(k0_mesh)
    
    silver = angled_sqrt(material.eps_ag(wfreq), bc_angle=0)
    surmof = angled_sqrt(material.eps_cav(wfreq), bc_angle=jnp.pi/2)

    ns = [silver, surmof, silver]
    ds = [0.02e-6, t_layer, 0.03e-6]

    # pcolormesh_k(K_r, K_i, jnp.squeeze(jnp.abs(ns[0])))
    # plt.colorbar()
    # plt.show()

    bc = 0.5*jnp.pi
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
    smat_mesh = stack(
        bc_angle=bc, bc_arcsin_i=bc, bc_arcsin_j=bc
    )

    num = len(t_layer)
    fig, axs = plt.subplots(3, int((num-0.1)//3+1))
    for idx, ax in enumerate(axs.flatten()):
        if idx == num:
            break
        plt.sca(ax)

        downsample = 20
        # z_j, f_j, w_j, z_n = aaa(
        #     k0_mesh[0, ::downsample, ::downsample], 
        #     smat_mesh[('in', 'out')][idx][::downsample, ::downsample], 
        #     tol=1e-7
        # )
        trans = jnp.abs(smat_mesh[('in', 'out')][idx])
        pcolormesh_k(K_r, K_i, trans, norm="log", vmin=1e-3, vmax=jnp.nanquantile(trans, 0.99))
        plt.colorbar()
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        # z_n = k_to_f(z_n)/1e12
        # plt.scatter(z_n.real, z_n.imag)
    plt.show()