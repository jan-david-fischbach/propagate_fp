from fresnel import fresnel_kx_direct
from propagation import propagation_kx
import sax

def stack_smat_kx(ds, ns, k0, kx, pol="s"):
    instances = {}
    connections = {}
    models = {
        "if": fresnel_kx_direct,
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