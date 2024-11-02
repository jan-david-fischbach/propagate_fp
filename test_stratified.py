# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import angled_stratified
import angled_stratified_treams
import fresnel
import sax
#%%
if __name__ == "__main__":
    ns = [2, 1.889+0.0035j, 1.802+0.0562j, 2.321+0.291j, 1.519+9j]
    # ns = [2, 1.889, 1.802, 2.321, 9j]
    ds = [1, 0.15, 0.03, 0.3, 0.2]

    # ns = [4, 2]
    # ds = [0.9, 1]

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

    stack, info = angled_stratified.stack_smat_kx(
        ds,
          ns, k0, kx, pol=pol, fresnel=fresnel.fresnel_kx_direct
    )
    smat_kx_direct = stack()

    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0, kx, pol=pol)
    smat_kx = stack()

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

    kx = jnp.linspace(0.1*k0, 0.7*k0, 7)
    k0_mesh = K_r+ 1j*K_i

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]

#%%
    smat_mesh = stack(
        k0=k0_mesh, kx=kx, 
        bc_angle=0.7*jnp.pi, bc_arcsin_i=0.7*jnp.pi, bc_arcsin_j=0.7*jnp.pi)

    from diffaaable import aaa
    for kx_idx in range(len(kx)):
        plt.figure()
        downsample = 20
        z_j, f_j, w_j, z_n = aaa(
            k0_mesh[0, ::downsample, ::downsample], 
            smat_mesh[('in', 'out')][kx_idx][::downsample, ::downsample], 
            tol=1e-7
        )

        plt.pcolormesh(K_r, K_i, jnp.abs(smat_mesh[('in', 'in')][kx_idx]), norm="log", vmin=0.1)
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.scatter(z_n.real, z_n.imag)
        plt.show()

# %%
