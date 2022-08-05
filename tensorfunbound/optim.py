import torch
from ipypb import ipb
import tensorfunbound as tfb
import tntorch as tn
import time

def optimize(
    model, dataloader, tdata, L=torch.dist, n_epochs=50, lrs=[1e-1, 1e-2, 1e-3, 1e-4]
):
    """
    model is a DL model, for ex. continuous wrapper around tensor train
    dataloader returns indices
    tdata is the original data array.

    in the learning
    y_goal = tdata[list(indices.T)]
    y_predicted = model(indices)
    """
    losses = []

    for lr in lrs:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        for epoch in ipb(range(n_epochs)):
            for x in dataloader:
                optimizer.zero_grad()
                indices = torch.stack(x, dim=1).squeeze().T
                y_goal = tdata[list(indices.T)]
                y_predicted = model(indices)
                loss = L(y_goal, y_predicted)
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append(loss.detach())
    return losses


def optimize_tntorch_style(
    tensors,
    loss_function,
    optimizer=torch.optim.Adam,
    tol=1e-4,
    max_iter=1e4,
    print_freq=250,
    verbose=True,
):
    """
    High-level wrapper for iterative learning.

    Default stopping criterion: either the absolute (or relative) loss improvement must fall below `tol`.
    In addition, the rate loss improvement must be slowing down.

    :param tensors: one or several tensors; will be fed to `loss_function` and optimized in place
    :param loss_function: must take `tensors` and return a scalar (or tuple thereof)
    :param optimizer: one from https://pytorch.org/docs/stable/optim.html. Default is torch.optim.Adam
    :param tol: stopping criterion
    :param max_iter: default is 1e4
    :param print_freq: progress will be printed every this many iterations
    :param verbose:
    """

    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    parameters = []
    for t in tensors:
        if isinstance(t, tn.Tensor):
            parameters.extend([c for c in t.cores if c.requires_grad])
            parameters.extend([U for U in t.Us if U is not None and U.requires_grad])
        elif t.requires_grad:
            parameters.append(t)
    if len(parameters) == 0:
        raise ValueError(
            "There are no parameters to optimize. Did you forget a requires_grad=True somewhere?"
        )

    optimizer = optimizer(parameters)
    losses = []
    converged = False
    start = time.time()
    iter = 0
    while True:
        optimizer.zero_grad()
        loss = loss_function(*tensors)
        if not isinstance(loss, (tuple, list)):
            loss = [loss]
        total_loss = sum(loss)
        total_loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(total_loss.detach())

        if len(losses) >= 2:
            delta_loss = torch.abs(losses[-1] - losses[-2])
        else:
            delta_loss = float("-inf")

        if iter >= 2 and tol is not None and (delta_loss <= tol):
            converged = True
            print("delta_loss = ", delta_loss)
            break
        if iter == max_iter:
            break
        if verbose and iter % print_freq == 0:
            print(
                "iter: {: <{}} | loss: ".format(iter, len("{}".format(max_iter))),
                end="",
            )
            print(" + ".join(["{:10.6f}".format(l.item()) for l in loss]), end="")
            if len(loss) > 1:
                print(" = {:10.4}".format(losses[-1].item()), end="")
            print(" | total time: {:9.4f}".format(time.time() - start))

        iter += 1
    if verbose:
        print("iter: {: <{}} | loss: ".format(iter, len("{}".format(max_iter))), end="")
        print(" + ".join(["{:10.6f}".format(l.item()) for l in loss]), end="")
        if len(loss) > 1:
            print(" = {:10.4}".format(losses[-1].item()), end="")
        print(" | total time: {:9.4f}".format(time.time() - start), end="")
        if converged:
            print(" <- converged (tol={})".format(tol))
        else:
            print(" <- max_iter was reached: {}".format(max_iter))


def optimize_pointwise_smooth(
    model,
    dataloader,
    tdata,
    true_dx=None,
    true_dy=None,
    L=torch.dist,
    n_epochs=100,
    lrs=[1e-1, 1e-2, 1e-3, 1e-4],
    c_local_grads=1e1,
    c_global_grads=1e-5
):
    """
    model is a DL model, for ex. continuous wrapper around tensor train
    dataloader returns indices
    tdata is the original data array.

    in the learning
    y_goal = tdata[list(indices.T)]
    y_predicted = model(indices)
    """
    # point-wise correspondence w.r.t. loss L is enforces locally
    # c1 enforces gradients locally (in points)
    # c_global_grads penalizes gradients globally
    
    global x
    losses = []


    for lr in lrs:
        optimizer = torch.optim.Adam(params=model.tt.cores, lr=lr)
        for epoch in ipb(range(n_epochs)):
            for x in dataloader:
                
                optimizer.zero_grad()
                indices = torch.stack(x, dim=1).squeeze().T
                y_goal = tdata[list(indices.T)]
                y_predicted = model(indices)

                
                loss = [L(y_goal, y_predicted) ]
                if c_local_grads>0: #only works if tdata is a matrix \ picture
                    goal_dx = true_dx[list(indices.T)]
                    goal_dy = true_dy[list(indices.T)]
                    grads = tfb.gradient(model.tt)
                    L1 = (goal_dx-grads[0][indices].torch() )**2 + \
                    (goal_dy-grads[1][indices].torch() )**2
                    loss+= [c_local_grads*sum(L1)]
                if c_global_grads>0:
#                     L2 = tn.partialset(model.tt).norm()
                    L2 = sum([ partial.norm() for partial in tfb.gradient(model.tt) ])
                    loss+= [c_global_grads*L2]
                
                loss = sum(loss)

                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append(loss.detach())
    return losses