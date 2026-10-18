
import numpy as np
import torch
import core


def gradcam(raw_win, class_index):
    """
    Generate a 1D Grad-CAM heatmap for a raw 1080-sample ECG window.
    Returns 1080 values normalized to [0, 1].
    """
    if class_index not in (0, 1, 2):
        raise ValueError("class_index must be 0, 1, or 2")

    model = core.get_model()
    model.eval()

    x = core.to_tensor(raw_win)
    activations = []
    gradients = []

    layer = model.f[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_index].backward()

        acts = activations[0]
        grads = gradients[0]

        weights = grads.mean(dim=2, keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))[0]

        cam = cam.detach().cpu().numpy()
        cam = np.interp(
            np.linspace(0, len(cam) - 1, core.WIN),
            np.arange(len(cam)),
            cam,
        )

        maximum = float(cam.max())
        if maximum > 0:
            cam = cam / maximum
        else:
            cam = np.zeros(core.WIN, dtype=np.float32)

        return cam.astype(np.float32)

    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)
