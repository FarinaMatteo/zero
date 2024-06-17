from .base import BaseTTAModule
from .zs import ZeroShot
from .zero import Zero
from .zero_rlcf import ZeroRLCF

__all__ = ["ZeroShot", "Zero", "ZeroRLCF"]

# Path: ttas/tpt.py
def get_tta_module(*args, **kwargs) -> BaseTTAModule:
    name = args[0]
    if name == "ZeroShot":
        return ZeroShot(*args, **kwargs)
    elif name == "Zero":
        return Zero(*args, **kwargs)
    elif name == "ZeroRLCF":
        return ZeroRLCF(*args, **kwargs)
    else:
        raise NotImplementedError(f"{name} not implemented.")