import jax.numpy as jnp

def propagation_i(ni=1, di=1, wl=1, theta_0=0, **kwargs):
    """Model the phase shift acquired as a wave propagates through medium A

    Args:
        ni: refractive index of medium (at wavelength wl)
        di: [μm] thickness of layer
        wl: [μm] wavelength
        theta: angle of incidence measured from normal in vacuum
    """
    k0 = 2*jnp.pi/(wl)
    kx = k0 * jnp.sin(theta_0)
    return propagation_kx(ni, di, k0, kx)

def angled_sqrt(x, bc_angle=jnp.pi):
    #return jnp.sqrt(x)
    arg = (bc_angle-jnp.pi)
    return jnp.sqrt(jnp.abs(x)) * jnp.exp(0.5j * (jnp.angle(x*jnp.exp(1j*arg)) - arg)) 

def propagation_kx(ni=1, di=1, k0=1, kx=0, bc_angle=jnp.pi, **kwargs):
    kz = angled_sqrt((k0*ni)**2-kx**2 + 0j, bc_angle)

    prop_i = jnp.exp(1j * kz * di)
    sdict = {
        ("left", "right"): prop_i,
        ("right", "left"): prop_i,
    }
    return sdict