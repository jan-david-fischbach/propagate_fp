#%%
import jax.numpy as jnp
from arcsin import arcsin
from propagation import angled_sqrt
from pprint import pprint
import sax

def debug_sdict(sdict):
    S, portmap = sax.sdense(sdict)
    #pprint(S)

def fresnel_mirror_ij(ni=1, nj=1, theta_0=0, pol="s", bc_arcsin_i=jnp.pi, bc_arcsin_j=jnp.pi, **kwargs):
    """Model a (fresnel) interface between two refractive indices

    Args:
        ni: refractive index of the initial medium
        nj: refractive index of the final
        theta: angle of incidence measured from normal in vacuum
        pol: "s" or "p" polarization
    """

    theta_i = arcsin(jnp.sin(theta_0)/ni, bc_arcsin=bc_arcsin_i)
    theta_j = arcsin(jnp.sin(theta_0)/nj, bc_arcsin=bc_arcsin_j) #need to investigate

    # theta_i = jnp.arcsin(jnp.sin(theta_0)/ni)
    # theta_j = jnp.arcsin(jnp.sin(theta_0)/nj)

    cos_i = jnp.cos(theta_i)
    cos_j = jnp.cos(theta_j)

    if pol in ["s", "TE"]:
        r_ij = (ni*cos_i-nj*cos_j) / (ni*cos_i + nj*cos_j)  
        # i->i reflection
        t_ij = 2*ni*cos_i / (ni*cos_i + nj*cos_j)  # i->j transmission
        t_ji = 2*nj*cos_j / (ni*cos_i + nj*cos_j) 
        
    elif pol in ["p", "TM"]:
        r_ij = (nj*cos_i-ni*cos_j) / (nj*cos_i + ni*cos_j)  
        # i->i reflection
        t_ij = 2*ni*cos_i / (nj*cos_i + ni*cos_j)  # i->j transmission
        t_ji = 2*nj*cos_j / (nj*cos_i + ni*cos_j)

    else:
        raise ValueError(f"polarization should be either 's'/'TM' or 'p'/'TE'")
    
    r_ji = -r_ij  # j -> i reflection
    
    sdict = {
        ("left", "left"): r_ij,
        ("left", "right"): t_ij,
        ("right", "left"): t_ji,
        ("right", "right"): r_ji,
    }
    # print(f"{ni=}, {nj=}:")
    debug_sdict(sdict)
    return sdict

def fresnel_kx(
        ni=1, nj=1, k0=1, kx=0, pol="s", 
        bc_arcsin=jnp.pi, bc_arcsin_i=jnp.pi, bc_arcsin_j=jnp.pi,
        **kwargs):
    theta_0 = arcsin(kx/k0 + 0j, bc_arcsin=bc_arcsin)
    return fresnel_mirror_ij(
        ni=ni, nj=nj, theta_0=theta_0, pol=pol, 
        bc_arcsin=bc_arcsin, bc_arcsin_i=bc_arcsin_i, bc_arcsin_j=bc_arcsin_j,
        **kwargs)

def fresnel_kx_direct(
    ni=1, nj=1, k0=1, kx=0, pol="p", bc_angle=jnp.pi, **kwargs
):

    kiz = -angled_sqrt((k0*ni)**2 - kx**2 + 0j, bc_angle)
    kjz = angled_sqrt((k0*nj)**2 - kx**2 + 0j, bc_angle)

    # print(f"{kiz=}; {kjz=}")
    if pol in ["s", "TE"]:
        eta = 1
    elif pol in ["p", "TM"]:
        eta = nj/ni
    else:
        raise ValueError(f"polarization should be either 's'/'TM' or 'p'/'TE'")
    
    r_ij = (eta*kiz+kjz/eta) / (eta*kiz-kjz/eta)
    t_ij = 2*kiz / (eta*kiz-kjz/eta)
    t_ji = (1-r_ij**2)/t_ij

    r_ji = -r_ij  # j -> i reflection

    sdict = {
        ("left", "left"): r_ij,
        ("left", "right"): t_ij,
        ("right", "left"): t_ji,
        ("right", "right"): r_ji,
    }
    # print(f"{ni=}, {nj=}:")
    debug_sdict(sdict)
    return sdict

#%%
if __name__ == "__main__":
    wl = 1.5
    k0 = 2*jnp.pi/wl
    theta_0=0.7
    kx = k0*jnp.sin(theta_0)
    pol = "s"
    n1 = 4
# %%
    fresnel_kx_direct(nj=n1, k0=k0, kx=kx, pol=pol)
# %%
    fresnel_mirror_ij(nj=n1, theta_0=theta_0, pol=pol)
# %%
