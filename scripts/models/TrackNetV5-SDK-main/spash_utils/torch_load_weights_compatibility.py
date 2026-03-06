import numpy as np
import _codecs
import torch
import os

def load_weights_compatibility():
    """
    Short description:
        Dark Magic happening here in order to fix a (randomly appearing) torch.load() exception.
        - When do the exception happens? 
            I don't know, if faced different behavior using the exact same docker image
        - When should I use it?
            Whenever you face an error similar to the one below.
        - Does it make any sense?
            Not much.
    
    More details:
        Going from torch 2.6 to 2.7, the serialization of the weights changed.
        By default torch.load() has load_weights=True to only load weights.
        This might cause issues when weights contains non-torch weights, and, for some reason,
        sometimes raises an error like:
        source of the fix, and discussion about that:
        https://github.com/MIC-DKFZ/nnUNet/issues/2681#issuecomment-2631266732

    Example trace this code is meant to fix:
    Command: 
        # on commit a0f5407e1549198422c832267959aa1561267f5c (2025-07-11)
        docker compose run --build --rm -it coello python3 -c "import torch; torch.load('models/osnet_x1_0-spash-padel-ep182.pt')"
    ```
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "/usr/local/lib/python3.12/dist-packages/boxmot/appearance/reid_auto_backend.py", line 47, in __init__
            self.model = self.get_backend()
                        ^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/boxmot/appearance/reid_auto_backend.py", line 74, in get_backend
            return backend_class(self.weights, self.device, self.half)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/boxmot/appearance/backends/pytorch_backend.py", line 13, in __init__
            super().__init__(weights, device, half)
        File "/usr/local/lib/python3.12/dist-packages/boxmot/appearance/backends/base_backend.py", line 35, in __init__
            self.load_model(self.weights)
        File "/usr/local/lib/python3.12/dist-packages/boxmot/appearance/backends/pytorch_backend.py", line 20, in load_model
            load_pretrained_weights(self.model, w)
        File "/usr/local/lib/python3.12/dist-packages/boxmot/appearance/reid_model_factory.py", line 150, in load_pretrained_weights
            checkpoint = torch.load(weight_path)
                        ^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1486, in load
            raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
        _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
                (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
                (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
                WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([scalar])` or the `torch.serialization.safe_globals([scalar])` context manager to allowlist this global if you trust this class/function.
    ```    
    """
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if np.__version__ == '1.24.4':
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
            np.dtype[np.float32],
            _codecs.encode
        ])
    else:
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.core.multiarray._reconstruct,
            np.dtype,
            np.dtype[np.float32],
            np.ndarray,
            np.dtypes.Float64DType,
            np.dtypes.Float32DType,
            _codecs.encode
        ])








