import jax.numpy as jnp
import numpy as onp
from scipy.constants import epsilon_0, c as c0, hbar
from fresnel import fresnel_mirror_ij, fresnel_kx
from propagation import propagation_i, propagation_kx
LOG=True

import sax
import meow
import meow.eme.propagate
sax_backend = meow.eme.sax._validate_sax_backend("klu")

def fields(padded_ns, zi, forwards, backwards, z, wl, theta_0=0, pol="s"):
    """
    Calculates the E-fields within a stack at given positions x
    
    Arguments
    ---------

    padded_ns: np.ndarray[complex]
        refractive index per layer
    zi: np.ndarray[float]
        position of the interface
        The first interface is assumed at 0, and should not be provided
    forwards: np.ndarray[complex]
        Complex amplitudes of the forward propagating waves
        specified at the left side of the layer
    backwards: np.ndarray[complex]
        Complex amplitudes of the backward propagating waves
        specified at the left side of the layer
    z: np.ndarray[float]
        Positions at which the E-field should be evaluated
    wl: float
        Wavelength of interest
    theta_0: float
        Incidence angle in vacuum given in rad
    pol: str
        Polarization either "s" or "p"

    

    """
    zi = onp.concatenate([[-onp.inf, 0], zi, [onp.inf]])
    E_tot = onp.zeros((len(z),), dtype=complex)
    Abs_tot = onp.zeros_like(z)
    k0 = 2*jnp.pi/(wl)
    kx = k0 * jnp.sin(theta_0)
    sign = 1 if pol=="s" else -1 
    # For adding up the forward and backward propagating waves
    # Related to the sign convention of the field components
    
    for n, forward, backward, z_min, z_max in zip(padded_ns, forwards, backwards, zi, zi[1:]):
        has_contribution = onp.any(onp.logical_and(z > z_min, z < z_max))
        if not has_contribution:
            continue

        i_min = onp.argmax(z >= z_min)
        i_max = onp.argmax(z > z_max)
        
        if i_max == 0:
            z_ = z[i_min:]
        else:
            z_ = z[i_min:i_max]

        if onp.isinf(z_min):
            z_local = z_
        else:
            z_local = z_ - z_min

        kz = jnp.sqrt((k0*n)**2-kx**2) #Wavenumber normal to the interfaces
        E_local = forward*onp.exp(1j * kz * z_local)
        E_local += sign*backward*onp.exp(-1j * kz * z_local)
        
        if i_max == 0:
            E_tot[i_min:] = E_local
        else:
            E_tot[i_min:i_max] = E_local

        eps = n**2 

        omega = 2*jnp.pi*c0/(wl*1e-6)
        Abs_local = 0.5* eps.imag * epsilon_0 * onp.abs(E_local)**2 * omega
        Abs_local /= hbar*omega #in number of photons

        from scipy.constants import h
       # Abs_local =  0.5 * eps.imag * onp.abs(E_local)**2 * 1 * omega * wl * 1e-6 * 1/(h*c0) * epsilon_0

        # print(f"{eps=}")
        # print(f"{omega=}")

        if i_max == 0:
            Abs_tot[i_min:] = Abs_local
        else:
            Abs_tot[i_min:i_max] = Abs_local
            
    return E_tot, Abs_tot

def split_square_matrix(matrix, idx):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix has to be square")
    return [matrix[:idx, :idx], matrix[:idx, idx:]], [
        matrix[idx:, :idx],
        matrix[idx:, idx:],
    ]

def propagate(l2rs, r2ls, excitation_l, excitation_r):
    forwards = []
    backwards = []
    for l2r, r2l in zip(l2rs, r2ls):
        s_l2r, p = sax.sdense(l2r)
        s_r2l, _ = sax.sdense(r2l)
        m = len([k for k in p.keys() if "right" in k])
        f, b = compute_mode_amplitudes(s_l2r, s_r2l, m, excitation_l, excitation_r)
        forwards.append(f)
        backwards.append(b)
    return forwards, backwards

def compute_mode_amplitudes(u, v, m, excitation_l, excitation_r):
    n = u.shape[0] - m
    l = v.shape[0] - m
    [u11, u21], [u12, u22] = split_square_matrix(u, n) 
    [v11, v21], [v12, v22] = split_square_matrix(v, m)
    #Sax uses notation where 1->2 is in entry 12.
    #The sandwich equations are derived for 1->2 in entry 21 (2 from 1)

    RHS = u21 @ excitation_l + u22 @ v12 @ excitation_r
    LHS = jnp.diag(jnp.ones(m)) - u22 @ v11
    forward = jnp.linalg.solve(LHS, RHS)
    backward = v12 @ excitation_r + v11 @ forward

    return forward, backward

def fields_in_stack(ds, ns, wl, theta_0, res=2000, pol="s"):
    ex_l = jnp.array([1])
    ex_r = jnp.array([0])

    identity = fresnel_mirror_ij(1, 1)

    propagations=[propagation_i(ni, di, wl, theta_0=theta_0) for ni, di in zip(ns, ds)]
    propagations=[identity]+propagations+[identity]
    propagations = {f"p_{i}": sax.sdense(p) for i, p in enumerate(propagations)}

    padded_ns = [1]+ns+[1]
    interfaces=[fresnel_mirror_ij(ni, nj, theta_0=theta_0, pol=pol) for ni, nj in zip(padded_ns, padded_ns[1:])]
    interfaces = {f"i_{i}_{i+1}": sax.sdense(p) for i, p in enumerate(interfaces)}

    pairs = meow.eme.propagate.pi_pairs(propagations, interfaces, sax_backend)
    l2rs = meow.eme.propagate.l2r_matrices(pairs, identity, sax_backend)
    r2ls = meow.eme.propagate.r2l_matrices(pairs, sax_backend)
    forwards, backwards = propagate(l2rs, r2ls, ex_l, ex_r)

    ds = jnp.array(ds)
    zi = jnp.cumsum(ds)
    z = jnp.linspace(0, jnp.round(jnp.sum(ds), 2), res)
    field, absorption = fields(padded_ns, zi, forwards, backwards, z, wl, 
                               theta_0=theta_0, pol=pol)
    
    T = jnp.abs(forwards[-1][0])**2
    R = jnp.abs(backwards[0][0])**2
    return z, field, absorption, T, R

def stack_smat(ds, ns, wl, theta_0, pol="s"):
    ns = [1]+ns+[1]
    instances = {}
    connections = {}
    models = {
        "if": fresnel_mirror_ij,
        "prop": propagation_i,
    }
    
    for i in range(len(ds)):
        settings = dict(ni=ns[i], nj=ns[i+1], wl=wl, 
                        theta_0=theta_0, di=ds[i], pol=pol)
        
        instances[f'if_{i}'] = {'component': 'if',     'settings': settings}
        settings = sax.update_settings(settings, ni=ns[i+1])
        instances[f'prop_{i}'] = {'component': 'prop', 'settings': settings}
        connections[f'if_{i},right'] = f'prop_{i},left'
        connections[f'prop_{i},right'] = f'if_{i+1},left'

    settings = dict(ni=ns[i+1], nj=ns[i+2], wl=wl, theta_0=theta_0, pol=pol)
    instances[f'if_{i+1}'] = {'component': "if", 'settings': settings}
    ports = {"in": "if_0,left", "out": f"if_{len(ds)},right"}

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports
    }

    return sax.circuit(
        netlist = netlist,
        models = models,
    )


def stack_smat_kx(ds, ns, k0, kx, pol="s", fresnel=fresnel_kx):
    ns = [1]+ns+[1]
    instances = {}
    connections = {}
    models = {
        "if": fresnel,
        "prop": propagation_kx,
    }
    
    for i in range(len(ds)):
        settings = dict(ni=ns[i], nj=ns[i+1], k0=k0, 
                        kx=kx, di=ds[i], pol=pol)
        
        instances[f'if_{i}'] = {'component': 'if',     'settings': settings}
        settings = sax.update_settings(settings, ni=ns[i+1])
        instances[f'prop_{i}'] = {'component': 'prop', 'settings': settings}
        connections[f'if_{i},right'] = f'prop_{i},left'
        connections[f'prop_{i},right'] = f'if_{i+1},left'

    settings = dict(ni=ns[i+1], nj=ns[i+2], k0=k0, kx=kx, pol=pol)
    instances[f'if_{i+1}'] = {'component': "if", 'settings': settings}
    ports = {"in": "if_0,left", "out": f"if_{len(ds)},right"}

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports
    }

    return sax.circuit(
        netlist = netlist,
        models = models,
    )