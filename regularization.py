# regularization.py
# Contains thin helpers for dropout and L2 usage; core logic is in forward_backward/backward
def apply_l2_to_cost(cost, parameters, lambd, m):
    """
    Adds L2 term to cost (if lambd>0), kept for clarity.
    """
    if lambd <= 0.0:
        return cost
    L2 = 0.0
    for l in range(1, len(parameters)//2 + 1):
        W = parameters["W" + str(l)]
        L2 += (lambd / (2.0 * m)) * (W**2).sum()
    return cost + L2
