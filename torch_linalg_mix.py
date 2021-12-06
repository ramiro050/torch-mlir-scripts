"""
Example of mixing the torch MLIR dialect with the linalg dialect.
"""

from typing import List

import torch
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

def _print_title(title: str):
    print()
    print(title)
    print('-' * len(title))

# Linalg operator definitions
@torch.jit.ignore
def init_tensor(sizes: List[int], dtype: int) -> torch.Tensor:
    return


@torch.jit.ignore
def matmul(m1: torch.Tensor, m2: torch.Tensor, out: torch.Tensor
           ) -> torch.Tensor:
    return


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.tanh(x)
        t = init_tensor([3,3], torch.float)
        return matmul(z, z, t)


# Compile and run Module
module = MyModule()
script_module = torch.jit.script(module)

print(script_module.graph)

mb = ModuleBuilder()
class_annotator = ClassAnnotator()
class_annotator.exportNone(script_module._c._type())
class_annotator.exportPath(script_module._c._type(), ["forward"])
class_annotator.annotateArgs(script_module._c._type(),
                             ["forward"],
                             [None,
                              ([3,3], torch.float32, True)])
mb.import_module(script_module._c, class_annotator)
mlir_module = mb.module

_print_title("Torch-MLIR")
mlir_module.dump()

# Compile the torch MLIR and execute the compiled program
with mlir_module.context:
    pipeline = ','.join(['torchscript-module-to-torch-backend-pipeline',
                         'torch-backend-to-linalg-on-tensors-backend-pipeline'])
    pm = PassManager.parse(pipeline)
pm.run(mlir_module)

_print_title("Linalg-MLIR")
print(mlir_module)

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(mlir_module)
jit_module = backend.load(compiled)

_print_title("Running Compiled Graph")
x = torch.randn((3,3), dtype=torch.float64)
print('Output from compiled MLIR:')
print(jit_module.forward(x.numpy()))
