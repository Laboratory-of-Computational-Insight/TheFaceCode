from typing import Optional

from utils.singleton.singlton import Singleton
import torch

DEFAULT_CONTEXT = "DEFAULT"


class Context:
    def __init__(self):
        self.cpu_device = torch.device("cpu")
        self.gpu_device = torch.device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__subcontext__ = set()

    class SubContext:
        def __init__(self, ctx, **kwargs):
            self.already_existing = {}
            self.values = kwargs
            self.ctx = ctx
            for k,v in kwargs.items():
                self.already_existing[k] = hasattr(ctx, k)

        def __enter__(self):
            for k in self.already_existing:
                if not self.already_existing[k]:
                    setattr(self.ctx, k, self.values[k])

        def __exit__(self, exc_type, exc_val, exc_tb):
            for k in self.already_existing:
                if not self.already_existing[k]:
                    delattr(self.ctx, k)



contexts = {DEFAULT_CONTEXT: Context()}


def get_context(ctx: Optional[str] = DEFAULT_CONTEXT):
    assert ctx in contexts, f"wanted context: {ctx} does not exist in contexts"
    return contexts[ctx]
