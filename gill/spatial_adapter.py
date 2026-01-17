"""
Spatial Adapter 统一入口模块

实现全部在 spatial_adapter_fixed.py（FP16 防溢出 clamp、GatedSelfAttention、inject 等）。
本文件仅做 re-export，训练/推理 `from gill.spatial_adapter import ...` 时实际跑的是 _fixed 里的代码。
"""

# 导入所有内容从 spatial_adapter_fixed（含 SpatialControlAdapter.forward 中的 FP16 clamp）
from .spatial_adapter_fixed import (
    SpatialPositionNet,
    SpatialAdapterModuleDict,
    SpatialControlProcessor,
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
    create_spatial_adapter_for_kolors,
    create_spatial_adapter_for_sdxl,
    load_spatial_adapter_state_dict,
)

# 导出所有公共接口
__all__ = [
    "SpatialPositionNet",
    "SpatialAdapterModuleDict",
    "SpatialControlProcessor",
    "inject_spatial_control_to_unet",
    "remove_spatial_control_from_unet",
    "create_spatial_adapter_for_kolors",
    "create_spatial_adapter_for_sdxl",
    "load_spatial_adapter_state_dict",
]
