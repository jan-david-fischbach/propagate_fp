# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import angled_stratified
import angled_stratified_treams
import fresnel
import sax
import jax
import psutil
from diffaaable import aaa
import time

# from jax_smi import initialise_tracking
# initialise_tracking()
#%%
if __name__ == "__main__":
    ns = [2, 1.889+0.0035j, 1.802+0.0562j, 2.321+0.291j, 1.519+9j]
    # ns = [2, 1.889, 1.802, 2.321, 9j]
    ds = [1, 0.15, 0.03, 0.3, 0.2]

    wl = 1.5
    theta_0=0.7#0.99 * jnp.pi/2
    k0 = 2*jnp.pi/wl
    kx = k0*jnp.sin(theta_0)

    pol = "p"
    pol_idx = 1 if pol=="p" else 0

    smat_treams = angled_stratified_treams.stack_smat(
        ds, ns, k0=k0, kx=kx, poltype="parity"
    )

    stack, info = angled_stratified.stack_smat(
        ds, ns, wl, theta_0=theta_0, pol=pol
    )
    smat = stack()

    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0, kx, pol=pol)
    smat_kx = stack()

    stack, info = angled_stratified.stack_smat_kx(
        ds,
          ns, k0, kx, pol=pol, fresnel=fresnel.fresnel_kx_direct
    )
    del info
    smat_kx_direct = stack()

    S          , portmap = sax.sdense(smat)
    S_kx       , portmap = sax.sdense(smat_kx)
    S_kx_direct, portmap = sax.sdense(smat_kx_direct)

    S_treams = jnp.array(smat_treams)[:,:,pol_idx, pol_idx][::-1]

    assert jnp.allclose(S, S_kx_direct)                 
    assert jnp.allclose(S_kx, S_kx_direct)
    assert jnp.allclose(S_treams, S_kx_direct)

# %%
    k_r = jnp.linspace(-0*k0, 1*k0, 300)
    k_i = jnp.linspace(-0.2*k0, 0.2*k0, 301)
    K_r, K_i = jnp.meshgrid(k_r, k_i)

    kx = jnp.linspace(0.1*k0, 0.7*k0, 200)
    k0_mesh = K_r+ 1j*K_i

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]
#%%
    stack = jax.jit(stack)
#%%
    for i in range(5):
        smat_mesh = stack(
            k0=k0_mesh, kx=kx, 
            bc_angle=0.7*jnp.pi, bc_arcsin_i=0.7*jnp.pi, bc_arcsin_j=0.7*jnp.pi)
        jax.block_until_ready(smat_mesh)
        time.sleep(1)
        print(f"Memory {i}:",psutil.Process().memory_info().rss / (1024 * 1024))
