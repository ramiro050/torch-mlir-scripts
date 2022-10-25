"""
Example of taking TorchScript nn Module and compiling it using torch-mlir.

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python torchscript_to_linalg.py`.
"""

import argparse

import torch
import torchvision.models as models
import torch_mlir
import warnings
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend

OUTPUT_TYPES = ['raw', 'torch', 'linalg-on-tensors']
parser = argparse.ArgumentParser()
parser.add_argument("--output-type", type=str, choices=OUTPUT_TYPES, default=None)
parser.add_argument("--check-with-eager", action="store_true")
args = parser.parse_args()

example_inputs = [torch.tensor([1]), torch.tensor([1])]
#placeholder_inputs = [torch_mlir.TensorPlaceholder.like(x, dynamic_axes=list(range(len(x.shape)))) for x in example_inputs]
placeholder_inputs = example_inputs

def foo(x):
    return (x,)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)

module = MyModule()
script_module = torch.jit.script(module, example_inputs)
if args.output_type is None:
    print(f"// OuputType: torchscript")
    print(script_module.graph)

for output_type in OUTPUT_TYPES:
    torch_mlir_module = torch_mlir.compile(script_module, placeholder_inputs,
                                           output_type=output_type)
    if args.output_type is None or output_type == args.output_type:
        print(f"// OuputType: {output_type}")
        print(torch_mlir_module)

if args.check_with_eager:
    backend = RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(torch_mlir_module)
    jit_module = backend.load(compiled)

    print("Running Compiled Graph")
    print('Expected output:')
    print(script_module.forward(*example_inputs))
    print('Output from compiled MLIR:')
    numpy_inputs = list(map(lambda x: x.numpy(), example_inputs))
    print(torch.tensor(jit_module.forward(*numpy_inputs)))

