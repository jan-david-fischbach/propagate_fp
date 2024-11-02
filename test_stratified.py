# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
#%%
if __name__ == "__main__":
    # ns = [2, 1.889+0.0035j, 1.802+0.0562j, 2.321+0.291j, 1.519+9j]
    # ns = [2, 1.889, 1.802, 2.321, 9j]
    # ds = [1, 0.15, 0.03, 0.3, 0.2]

    ns = [4]
    ds = [0.9]

    wl = 1.5
    theta_0=1#0.99 * jnp.pi/2

    # field, absorption, x = stratified.multilayer_stack(ns, ds)

    # plt.plot(x, jnp.abs(field))
    # plt.axhline(0.13)
    # #plt.axhline(1, color="k", linestyle="--")
    # plt.ylim((0, None))
    # plt.xlabel("x [$\mu$m]")
    # plt.ylabel("$|E(x)|$")
    # plt.show()
# %%

    import angled_stratified
    import angled_stratified_treams
    import fresnel
    pol = "p"

    # stack, info = angled_stratified.stack_smat(ds, ns, wl, theta_0=theta_0, pol=pol)
    # smat = stack()

    k0 = 2*jnp.pi/wl
    kx = k0*jnp.sin(theta_0)

    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0, kx, pol=pol)
    smat_kx = stack()

    print("#################")
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0, kx, pol=pol, fresnel=fresnel.fresnel_kx_direct)
    smat_kx_direct = stack()

    smat_treams = angled_stratified_treams.stack_smat(ds, ns, k0=k0, kx=kx, poltype="parity")
# %%
    #assert jnp.allclose(smat_kx[('in', 'in' )], smat_kx_direct[('in', 'in')])
    
    pol_idx = 0 if pol=="p" else 1
    assert jnp.allclose(smat_kx_direct[('in', 'out')], jnp.array(smat_treams[0,0])[pol_idx,pol_idx])
    assert jnp.allclose(smat_kx_direct[('in', 'in')],  -jnp.array(smat_treams[1,0])[pol_idx,pol_idx])

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
        k0=k0_mesh, kx=kx, bc_angle=0.7*jnp.pi)

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
    #%%
    # for k0_test in jnp.squeeze(k0_mesh).flatten():
    #     for kx_test in jnp.squeeze(kx):
    #         stack_smat(ds, ns, k0=k0_test, kx=kx_test, poltype="parity")

# %%
