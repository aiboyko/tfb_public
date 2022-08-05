def eikonal2d(mask, goal, speed=None):
    if speed is None:
        speed = np.ones_like(mask)

    ix_goal, iy_goal = goal

    if not mask[ix_goal, iy_goal]:
        phi = np.ones_like(mask)
        phi[ix_goal, iy_goal] = -1
        phi = np.ma.MaskedArray(phi, mask)
        try:
            fulldat = skfmm.travel_time(phi, speed, dx=1e-2)
        except:
            pass
    else:
        print(ix_goal, iy_goal)
        print("is inside mask")
        fulldat = ma.MaskedArray(data=speed, mask=speed.astype("bool"))

    return torch.Tensor(fulldat.data * ~fulldat.mask).numpy(), fulldat
