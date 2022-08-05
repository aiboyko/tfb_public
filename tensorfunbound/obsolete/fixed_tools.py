# replace the function in the tools.py
def mask(t, mask):
    """
    Masks a tensor. Basically an element-wise product, but this function makes sure slices are matched according to their "meaning" (as annotated by the tensor's `idx` field, if available)

    :param t: input :class:`Tensor`
    :param mask: a mask :class:`Tensor`

    :return: masked :class:`Tensor`
    """
    device = t.cores[0].device
    if not hasattr(t, "idxs"):
        idxs = [np.arange(sh) for sh in t.shape]
    else:
        idxs = t.idxs
    cores = []
    Us = []
    for n in range(t.dim()):
        idx = np.array(idxs[n])
        idx[idx >= mask.shape[n]] = mask.shape[n] - 1  # Clamp
        if mask.Us[n] is None:
            cores.append(mask.cores[n][..., idx, :].to(device))
            Us.append(None)
        else:
            cores.append(mask.cores[n].to(device))
            Us.append(mask.Us[n][idx, :])
    mask = tn.Tensor(cores, Us, device=device)
    return t * mask
