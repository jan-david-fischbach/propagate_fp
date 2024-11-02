import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, c as c0

import sax
import meow
import meow.eme.propagate
sax_backend = meow.eme.sax._validate_sax_backend("klu")


def fresnel_mirror_ij(ni=1.0, nj=1.0):
    """Model a (fresnel) interface between twoo refractive indices

    Args:
        ni: refractive index of the initial medium
        nf: refractive index of the final
    """
    r_fresnel_ij = (ni - nj) / (ni + nj)  # i->j reflection
    t_fresnel_ij = 2 * ni / (ni + nj)  # i->j transmission
    r_fresnel_ji = -r_fresnel_ij  # j -> i reflection
    t_fresnel_ji = (1 - r_fresnel_ij**2) / t_fresnel_ij  # j -> i transmission 2 * nj / (ni + nj)#
    sdict = {
        ("left", "left"): r_fresnel_ij,
        ("left", "right"): t_fresnel_ij,
        ("right", "left"): t_fresnel_ji,
        ("right", "right"): r_fresnel_ji,
    }
    return sdict


def propagation_i(ni=1.0, di=0.5, wl=1.0):
    """Model the phase shift acquired as a wave propagates through medium A

    Args:
        ni: refractive index of medium (at wavelength wl)
        di: [μm] thickness of layer
        wl: [μm] wavelength
    """
    prop_i = jnp.exp(1j * 2 * jnp.pi * ni * di / wl)
    sdict = {
        ("left", "right"): prop_i,
        ("right", "left"): prop_i,
    }
    return sdict

def multilayer_stack(ns, ds, wl=1.0, ex_l=1, ex_r=0, x=jnp.linspace(-0.4, 2, 400)):
    identity = fresnel_mirror_ij(1, 1)
    xi = jnp.cumsum(jnp.array(ds))

    propagations=[propagation_i(ni, di, wl) for ni, di in zip(ns, ds)]
    propagations=[identity]+propagations+[identity]
    propagations = {f"p_{i}": sax.sdense(p) for i, p in enumerate(propagations)}

    padded_ns = [1]+ns+[1]
    interfaces=[fresnel_mirror_ij(ni, nj) for ni, nj in zip(padded_ns, padded_ns[1:])]
    #interfaces=[identity]+interfaces+[identity]
    interfaces = [sax.sdense(i) for i in interfaces]
    interfaces = {f"i_{i}_{i+1}": (s.T, p) for i, (s, p) in enumerate(interfaces)}

    pairs = meow.eme.propagate.pi_pairs(propagations, interfaces, sax_backend)
    l2rs = meow.eme.propagate.l2r_matrices(pairs, identity, sax_backend)
    r2ls = meow.eme.propagate.r2l_matrices(pairs, sax_backend)

    ex_l = jnp.array([ex_l])
    ex_r = jnp.array([ex_r])

    forwards, backwards = meow.eme.propagate.propagate(l2rs, r2ls, ex_l, ex_r)
    field, absorption = fields(padded_ns, xi, forwards, backwards, x, wl)

    return field, absorption, x


def fields(padded_ns, xi, forwards, backwards, x, wl):
    """Calculates the fields within a stack at given positions x
    Attention xi > 0
    """
    xi = onp.concatenate([[-onp.inf, 0], xi, [onp.inf]])
    E_tot = onp.zeros((len(x),), dtype=complex)
    Abs_tot = onp.zeros((len(x),), dtype=complex)
    for n, forward, backward, x_min, x_max in zip(padded_ns, forwards, backwards, xi, xi[1:]):
        has_contribution = onp.any(onp.logical_and(x > x_min, x < x_max))
        if not has_contribution:
            continue
        print(f"{n}: {x_min} - {x_max}: -> {forward[0]}; <- {backward[0]}")
        i_min = onp.argmax(x >= x_min)
        i_max = onp.argmax(x > x_max)
        
        if i_max == 0:
            x_ = x[i_min:]
        else:
            x_ = x[i_min:i_max]

        if onp.isinf(x_min):
            x_local = x_
        else:
            x_local = x_ - x_min
        E_local = forward*onp.exp(2j * onp.pi * n / wl * x_local)
        E_local += backward*onp.exp(-2j * onp.pi * n / wl * x_local)

        if i_max == 0:
            E_tot[i_min:] = E_local
        else:
            E_tot[i_min:i_max] = E_local

        eps = n**2 * epsilon_0
        omega = 2 * onp.pi * c0/wl
        Abs_local = 0.5* omega * eps.imag * onp.abs(E_local)**2
        if i_max == 0:
            Abs_tot[i_min:] = Abs_local
        else:
            Abs_tot[i_min:i_max] = Abs_local
            
    return E_tot, Abs_tot
