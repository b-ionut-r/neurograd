1. PAD layer: if memsave is on, after forward, modifies .data attribute of
input Tensor to reference the inner part of output's .data
2. SlidingWindowView: produces view, accumulates gradient backwards though intelligent view buffer overwriting.
3. Since all differentiable OPS of the Neurograd Framework are
single-outputed, during the Tensor.backwards() reversed topological order traversation, after accumulating gardients to the parent tensors that depend on non-leaf/non-terminal tensor, its .data and .grad xp.ndarrays attributes are dereferenced. This way GC will kick in and Cupy will eventually free up memory. This tensors are intermediary and
therefore we assume we wont use them later. Enabled by default with retain_graph=False. Also, to avoid dereferencing useful 
last tensors, like y_true, y_pred, loss, scaled_loss we also
have preserve_ancestors=4 (4 last topo layers keep their .data intact)
4. BatchNormalizer OPS: if memsave is on, after forward, it dereferences input.data and assigns it output.data, which can be used in backwards pass to reconstruct x_hat, used in grads
computation.
