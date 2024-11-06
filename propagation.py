import jax.numpy as jnp

def angled_sqrt(x, bc_angle=jnp.pi, nan_tolerance=0.01*jnp.pi):
    arg = (bc_angle-jnp.pi)
    adjusted_angle = jnp.angle(x*jnp.exp(1j*arg))

    adjusted_angle = jnp.where(
        jnp.abs(adjusted_angle)<=(jnp.pi-nan_tolerance), 
        adjusted_angle, 
        jnp.nan)
    
    return jnp.sqrt(jnp.abs(x)) * jnp.exp(0.5j * (adjusted_angle - arg)) 

def propagation_kx(ni=1, di=1, k0=1, kx=0, bc_angle=jnp.pi/2, **kwargs):
    kz = angled_sqrt((k0*ni)**2-kx**2 + 0j, bc_angle, nan_tolerance=0)

    prop_i = jnp.exp(1j * kz * di)
    sdict = {
        ("left", "right"): prop_i,
        ("right", "left"): prop_i,
    }
    return sdict