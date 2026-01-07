"""
Spatial Adapter 统一入口模块

为了兼容性，将 spatial_adapter_fixed.py 的内容导出为 spatial_adapter
"""

# 导入所有内容从 spatial_adapter_fixed
from .spatial_adapter_fixed import (
    SpatialPositionNet,
    SpatialAdapterModuleDict,
    SpatialControlProcessor,
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
    create_spatial_adapter_for_kolors,
    create_spatial_adapter_for_sdxl,
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
]
