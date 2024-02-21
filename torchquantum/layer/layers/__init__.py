from .module_from_ops import QuantumModuleFromOps
from .op_all import TrainableOpAll, ClassicalInOpAll, FixedOpAll, TwoQAll
from .random_layers import RandomLayer, RandomLayerAllTypes, RandomOp1All
from .swap_layer import SWAPSWAPLayer, SWAPSWAPLayer0
from .cx_layer import CXLayer, CXCXCXLayer
from .ry_layer import (
    RYRYCXLayer0,
    RYRYRYCXCXCXLayer0,
    RYRYRYLayer0,
    RYRYRYSWAPSWAPLayer0,
)
from .u3_layer import U3CU3Layer0, CU3Layer0
from .qft_layer import QFTLayer
from .seth_layer import SethLayer0, SethLayer1, SethLayer2
from .layers import (
    SimpleQLayer,
    Op1QAllLayer,
    LayerTemplate0,
    CXRZSXLayer0,
    RZZLayer0,
    BarrenLayer0,
    FarhiLayer0,
    MaxwellLayer0,
    RXYZCXLayer0,
)

# layer (children of LayerTemplate0) to add to the layer_name_dict
_all_layers = [
    SWAPSWAPLayer0,
    RYRYCXLayer0,
    RYRYRYCXCXCXLayer0,
    RYRYRYLayer0,
    RYRYRYSWAPSWAPLayer0,
    CXRZSXLayer0,
    RZZLayer0,
    BarrenLayer0,
    FarhiLayer0,
    MaxwellLayer0,
    RXYZCXLayer0,
]

layer_name_dict = {}

for _lyr in _all_layers:
    # check the layer has a non-empty name
    assert _lyr.name is not None, f"Layer name not defined for {layer}"
    layer_name_dict[_lyr.name] = _lyr
