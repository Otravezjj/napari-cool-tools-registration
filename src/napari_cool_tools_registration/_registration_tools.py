"""
This module contains code for registering volumetric and image data.
"""
import numpy as np
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_img_proc import torch, viewer, device
from napari_cool_tools_img_proc._normalization import normalize_data_in_range_pt_func

def a_scan_correction(vol:Image):
    """"""
    a_scan_correction_thread(vol=vol)

    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def a_scan_correction_thread(vol:Image) -> Layer:
    ''''''
    show_info(f'A-scan correction thread has started')
    layer = a_scan_correction_func(vol=vol)
    show_info(f'A-scan correction thread has completed')

    return layer

def a_scan_correction_func(vol:Image) -> Layer:
    """"""
    data = vol.data
    data = normalize_data_in_range_pt_func(data, 0.0, 1.0)
    name = vol.name
    vol.name = f"{vol.name}_pre_ascan_correction"
    h = data.shape[0]
    d = data.shape[1]
    w = data.shape[2]

    Xn = np.arange(w)
    x_org = (w/2)*np.sin(2*np.pi/(2*w)*Xn-np.pi/2) + (w/2)

    vol_out = np.empty_like(data)

    for i in tqdm(range(h),desc="Complete"):
        for j in range(d):
            vol_out[i,j,:] = np.interp(Xn,x_org,data[i,j,:])

    vol_out = normalize_data_in_range_pt_func(vol_out, 0.0, 1.0)

    add_kwargs = {"name":name}
    layer_type = "image"
    layer = Layer.create(vol_out,add_kwargs,layer_type)

    print(f"data min,max: ({data.min()},{data.max()}), vs vol_out min,max: ({vol_out.min()},{vol_out.max()})\n")

    return layer