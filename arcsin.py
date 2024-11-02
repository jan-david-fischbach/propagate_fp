#%%
import jax.numpy as jnp
import matplotlib.pyplot as plt

from propagation import angled_sqrt

def arcsin(z, bc_arcsin=jnp.pi):
    return - 1j* jnp.log(1j*z + angled_sqrt(1-z**2, bc_arcsin))


if __name__ == "__main__":
    kx = 1

    z_r = jnp.linspace(-3, 3, 200)
    z_i = jnp.linspace(-3, 3, 201)
    Z_r, Z_i = jnp.meshgrid(z_r, z_i)
    Z = Z_r + 1j*Z_i
    #%%
    plt.pcolormesh(Z_r, Z_i, jnp.imag(arcsin(Z, bc_arcsin=1.4*jnp.pi)))

    #%%
    plt.pcolormesh(Z_r, Z_i, jnp.imag(jnp.arcsin(Z)))    
    # %%

    plt.pcolormesh(Z_r, Z_i, jnp.abs(arcsin(kx/Z, bc_arcsin=1.4*jnp.pi))) 
# %%
