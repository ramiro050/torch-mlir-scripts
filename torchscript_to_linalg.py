"""
Example of taking TorchScript nn Module and compiling it using torch-mlir.

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python torchscript_to_linalg.py`.
"""

import torch
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

from torchfx.builder import build_module

def _print_title(title: str):
    print()
    print(title)
    print('-' * len(title))

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.add(x, y)

module = MyModule()
script_module = torch.jit.script(module)

mb = ModuleBuilder()
class_annotator = ClassAnnotator()
class_annotator.exportNone(script_module._c._type())
class_annotator.exportPath(script_module._c._type(), ["forward"])
class_annotator.annotateArgs(script_module._c._type(),
                             ["forward"],
                             [None,
                              ([-1, -1], torch.float32, True),
                              ([-1, -1], torch.float32, True)])
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
x = torch.randn((3, 4), dtype=torch.float)
y = torch.randn((3, 4), dtype=torch.float)
print('Expected output:')
print(script_module.forward(x, y))
print('Output from compiled MLIR:')
print(jit_module.forward(x.numpy(), y.numpy()))
