"""
To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

Call `python bert_torchscript_to_linalg.py`.
"""

from typing import List

import torch
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from annotations import forward_annotations, backward_annotations

def _print_title(title: str):
    print()
    print(title)
    print('-' * len(title))

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        #t = init_tensor([3,3], torch.float)
        return torch.argmax(x, 0, keepdim=True)

script_module = torch.jit.load("/home/ramiroleal/forwardModule.pt")

print(script_module.graph)

mb = ModuleBuilder()
class_annotator = ClassAnnotator()
class_annotator.exportNone(script_module._c._type())
class_annotator.exportPath(script_module._c._type(), ["forward"])
class_annotator.annotateArgs(script_module._c._type(),
                             ["forward"],
                             forward_annotations)
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
