"""
Example of taking TorchScript nn Module and compiling it using torch-mlir.

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python torchscript_to_linalg.py`.
"""

import torch
import torchvision.models as models
import torch_mlir
import warnings
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend

def _print_title(title: str):
    print()
    print(title)
    print('-' * len(title))

example_inputs = [torch.rand((3,4,5,6)), torch.randint(0, 1, (4,2)), torch.tensor([0, 1])]
placeholder_inputs = [torch_mlir.TensorPlaceholder.like(x, dynamic_axes=[]) for x in example_inputs]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        return torch.ops.aten.index(a, [b, None, c])


module = MyModule()
script_module = torch.jit.script(module, example_inputs)
_print_title("TorchScript")
print(script_module.graph)

torch_module = torch_mlir.compile(script_module, placeholder_inputs,
                                  output_type=torch_mlir.OutputType.RAW)
_print_title("Torch-MLIR")
print(torch_module)

torch_module = torch_mlir.compile(script_module, placeholder_inputs,
                                  output_type=torch_mlir.OutputType.TORCH)
_print_title("Torch-MLIR-backend")
print(torch_module)

linalg_module = torch_mlir.compile(script_module, placeholder_inputs,
                                   output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

_print_title("Linalg-MLIR")
print(linalg_module)

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(linalg_module)
jit_module = backend.load(compiled)

_print_title("Running Compiled Graph")
print('Expected output:')
print(script_module.forward(*example_inputs))
print('Output from compiled MLIR:')
numpy_inputs = list(map(lambda x: x.numpy(), example_inputs))
print(torch.tensor(jit_module.forward(*numpy_inputs)))
