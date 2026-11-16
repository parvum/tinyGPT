"""
Muon optimizer (MomentUm Orthogonalized by Newton-Schulz), as in modded-nanogpt / nanochat.

For a weight matrix W with gradient G:
    1. momentum, like SGD:           buf = 0.95 * buf + 0.05 * G   (+ Nesterov look-ahead)
    2. orthogonalize the update:      buf = U S V^T  ->  update = U V^T
       every direction of the matrix gets the same step size (all singular values set to 1),
       instead of the few strongest directions taking almost all of it
    3. W -= lr * update

U V^T is found with a few Newton-Schulz iterations (only matrix multiplies, no SVD).
Muon only makes sense for 2D hidden->hidden matrices. Embeddings, lm_head, norms, biases and
the tiny MoE router stay on AdamW, so training uses two optimizers side by side.
"""
import torch


@torch.no_grad()
def orthogonalize(G, steps=5):
    # Newton-Schulz: X <- a X + b (X X^T) X + c (X X^T)^2 X. This odd polynomial keeps U and V and moves
    # every singular value towards ~1 (the coefficients are tuned for speed, so it lands in ~0.7..1.2)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T # work with the wide shape, so X X^T is the smaller matrix
    X = X / (X.norm() + 1e-7) # all singular values <= 1, where the iteration converges
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue # e.g. an MoE expert that no token was routed to
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - group['momentum']) # buf = momentum * buf + (1 - momentum) * g
                g = g.lerp(buf, group['momentum']) if group['nesterov'] else buf
                update = orthogonalize(g, group['ns_steps'])
                # U V^T of a tall matrix has smaller entries, scale so the step size does not depend on the shape
                update *= max(1, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update.to(p.dtype), alpha=-group['lr'])


def setup_optimizers(model, name='adamw', lr=6e-4, muon_lr=0.02, weight_decay=0.0):
    # returns a list of optimizers, the training loop calls zero_grad / step on each
    if name == 'adamw':
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)]
    assert name == 'muon', name
    muon_params, adam_params = [], []
    for pname, p in model.named_parameters():
        is_hidden_matrix = p.ndim == 2 and pname.startswith('blocks.') and '.router.' not in pname
        (muon_params if is_hidden_matrix else adam_params).append(p)
    n_muon = sum(p.numel() for p in muon_params)
    n_adam = sum(p.numel() for p in adam_params)
    print(f'Muon: {len(muon_params)} matrices, {n_muon:,} params | AdamW: {len(adam_params)} tensors, {n_adam:,} params')
    return [Muon(muon_params, lr=muon_lr),
            torch.optim.AdamW(adam_params, lr=lr, weight_decay=weight_decay)]
