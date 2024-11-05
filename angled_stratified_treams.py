import treams
def stack_smat(ds, ns, k0, kx, poltype=None):
    ns = [1] + ns + [1]
    pwb = treams.PlaneWaveBasisByComp([[kx, 0, 0],[kx, 0, 1]])
    stack = []

    for i in range(len(ds)):
        inter = treams.SMatrices.interface(pwb, k0, [ns[i]**2, ns[i+1]**2], poltype=poltype)
        prop  = treams.SMatrices.propagation([0,0,ds[i]], pwb, k0, [ns[i+1]**2], poltype=poltype)
        stack.append(inter)
        stack.append(prop)
        
    inter = treams.SMatrices.interface(pwb, k0, [ns[-2]**2, ns[-1]**2], poltype=poltype)
    stack.append(inter)

    return treams.SMatrices.stack(stack)