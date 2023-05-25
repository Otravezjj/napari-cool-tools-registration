"""
This module contains code for registering volumetric and image data.
"""
import numpy as np
from typing import Dict,List
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch, viewer
from napari_cool_tools_img_proc._normalization import normalize_data_in_range_pt_func
from napari_cool_tools_segmentation._segmentation import memory_stats

def a_scan_correction(vol:Image):
    """"""
    a_scan_correction_thread(vol=vol)

    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def a_scan_correction_thread(vol:Image) -> Layer:
    ''''''
    show_info(f'A-scan correction thread has started')
    layer = a_scan_correction_func(vol=vol)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'A-scan correction thread has completed')

    return layer

def a_scan_correction_func(vol:Image) -> Layer:
    """"""
    data = vol.data
    data = normalize_data_in_range_pt_func(data, 0.0, 1.0)
    name = f"{vol.name}_AS_Corr"
    #vol.name = f"{vol.name}_pre_ascan_correction"
    h = data.shape[0]
    d = data.shape[1]
    w = data.shape[2]

    Xn = np.arange(w)
    x_org = (w/2)*np.sin(2*np.pi/(2*w)*Xn-np.pi/2) + (w/2)

    vol_out = np.empty_like(data)

    for i in tqdm(range(h),desc="A-scan Correction"):
        for j in range(d):
            vol_out[i,j,:] = np.interp(Xn,x_org,data[i,j,:])

    vol_out = normalize_data_in_range_pt_func(vol_out, 0.0, 1.0)

    add_kwargs = {"name":name}
    layer_type = "image"
    layer = Layer.create(vol_out,add_kwargs,layer_type)

    #print(f"data min,max: ({data.min()},{data.max()}), vs vol_out min,max: ({vol_out.min()},{vol_out.max()})\n")

    return layer

def a_scan_reg_calc_settings(vol:Image, min_regs:int=3, max_regs:int=8):
    """Calculate optimal number of regions to divide volume into for a-scan registration.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        max_regs (int): maximum number of regions that will be considered for a-scan registration

    Returns:
        int indicating optimal number of regions to use when performing a-scan registration
    """
    def display_settings(settings):
        show_info(f"Optimal number of regions: {settings['region_num']}")
        show_info(f"Avg phase difference: {settings['avg_phase_diff']}")
        return
    
    worker = a_scan_reg_calc_settings_thread(vol=vol,min_regs=min_regs,max_regs=max_regs)
    worker.returned.connect(display_settings)
    worker.start()

@thread_worker(progress=True)
def a_scan_reg_calc_settings_thread(vol:Image, min_regs:int=3, max_regs:int=8) -> Dict:
    """Calculate optimal number of regions to divide volume into for a-scan registration.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        max_regs (int): maximum number of regions that will be considered for a-scan registration

    Returns:
        int indicating optimal number of regions to use when performing a-scan registration
    """
    show_info(f'A-scan region calc thread has started')
    ascan_settings = a_scan_reg_calc_settings_func(vol=vol,min_regs=min_regs,max_regs=max_regs)
    show_info(f'A-scan region calc thread has completed')

    return ascan_settings

def a_scan_reg_calc_settings_func(vol:Image, min_regs:int=3, max_regs:int=8) -> Dict:
    """Calculate optimal number of regions to divide volume into for a-scan registration.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        max_regs (int): maximum number of regions that will be considered for a-scan registration

    Returns:
        int indicating optimal number of regions to use when performing a-scan registration
    """

    from skimage.registration import phase_cross_correlation

    data = vol.data
    #name = vol.name

    ascan_settings = {"region_num":None,"regions":None,"shifts":None,"avg_phase_diff":None}

    for sections in tqdm(range(min_regs,max_regs+1),desc="Testing number of regions\n"):

        sec_len = int(data.shape[0] / sections)
        
        regions = []
        shifts = []
        phase_diffs = []

        for s in tqdm(range(sections),desc="Creating Regions\n"):
            start = s*sec_len
            end = start + sec_len
            curr = data[:,:,start:end]
            regions.append(curr)

            even = curr[::2,:,:]
            odd = curr[1::2,:,:]
            
            shift, error, diffphase = phase_cross_correlation(even, odd, upsample_factor=100)
            print(f"\n\nshift:{shift}\nerror: {error}\ndiffphase: {diffphase}\n\n")
            shifts.append(shift[2])
            phase_diffs.append(diffphase)

        avg_phase_diff = np.array(phase_diffs)
        avg_phase_diff = np.absolute(avg_phase_diff)
        avg_phase_diff = np.mean(avg_phase_diff,axis=0)
        avg_phase_diff = avg_phase_diff

        print(f"\n\nSection region(s) has/have an avg phase diff of {avg_phase_diff}\n\n")

        if sections == min_regs:
            ascan_settings["region_num"] = sections
            ascan_settings["regions"] = regions
            ascan_settings["shifts"] = shifts
            ascan_settings["avg_phase_diff"] = avg_phase_diff
        else:
            if ascan_settings["avg_phase_diff"] > avg_phase_diff:
                print(f"Improved phase diff from {ascan_settings['avg_phase_diff']} to {avg_phase_diff}\n")
                print(f"Optimal number of regions has changed from {ascan_settings['region_num']} to {sections}\n")
                ascan_settings["region_num"] = sections
                ascan_settings["regions"] = regions
                ascan_settings["shifts"] = shifts
                ascan_settings["avg_phase_diff"] = avg_phase_diff
                
            else:
                pass

    return ascan_settings

def a_scan_reg_subpix(vol:Image, sections:int=4, sub_pixel_threshold:float=0.5, fill_gaps=True, roll_over_flag=True):
    ''''''
    a_scan_reg_subpix_thread(vol=vol,sections=sections,sub_pixel_threshold=sub_pixel_threshold,fill_gaps=fill_gaps,roll_over_flag=roll_over_flag)

    return

@thread_worker(connect={"yielded": viewer.add_layer},progress=True)
def a_scan_reg_subpix_thread(vol:Image, sections:int=4, sub_pixel_threshold:float=0.5, fill_gaps=True,roll_over_flag=True) -> List[Layer]:
    ''''''

    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    show_info(f'A-scan registration thread has started')

    #sections = 4
    data = vol.data
    name = vol.name

    replace_val = round(data.max()+2)

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_ascan_subpix_reg"}

    # optional layer type argument
    layer_type = "image"

    even = data[::2,:,:]
    odd = data[1::2,:,:]

    sec_len = int(data.shape[0] / sections)

    regions = []

    for s in tqdm(range(sections),desc="Creating Regions\n"):
        start = s*sec_len
        end = start + sec_len
        curr = data[:,:,start:end]
        regions.append(curr)

    shifts = []

    for r in tqdm(regions,desc="Calculating Shifts for region\n"):
        even = r[::2,:,:]
        odd = r[1::2,:,:]
        shift, error, diffphase = phase_cross_correlation(even, odd, upsample_factor=100)
        print(f"\n\nshift:{shift}\nerror: {error}\ndiffphase: {diffphase}\n\n")
        shifts.append(shift[2])

    out_reg = []
    roll_overs = []

    for i,s in tqdm(enumerate(shifts),desc="Shifting Regions\n"):
        shift2 = round(s)
        shift_idx = abs(shift2)

        out = np.empty_like(regions[i])
        out[::2,:,:] = regions[i][::2,:,:]

        input_ = np.fft.fft2(regions[i][1::2,:,:])
        result = fourier_shift(input_,(0.0,0.0,s),axis=2)
        result = np.fft.ifft2(result)
        odd_out = result.real

        if s < 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,-shift_idx:].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,-shift_idx:] = replace_val

                
                if i > 0 and shifts[i-1] < 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                    out_reg_odd = out_reg[i-1][1::2,:,:]

                    axis_0_len,axis_1_len = out_reg_odd.shape[0], out_reg_odd.shape[1]

                    print(f"\nout_reg_odd shape: {out_reg_odd.shape}\n")

                    print(f"\nout_reg_odd[{i-1}] range: ({out_reg_odd.min()}, {out_reg_odd.max()})\n")

                    gap_idx = (out_reg_odd == replace_val)

                    gap = out_reg_odd[gap_idx]

                    gap.shape = (axis_0_len,axis_1_len,-1)

                    print(f"\nout_reg gap shape: {gap.shape}\nroll_over shape: {roll_over.shape}\n")

                    gap_roll_diff = gap.shape[2] - roll_over.shape[2]

                    if roll_over_flag:
                        if gap_roll_diff >= 0:
                            out_reg[i-1][1::2,:,-roll_over.shape[2]:] = roll_over
                            print(f"\npositive or neutral gap_roll_diff s < 0\n")
                        elif gap_roll_diff < 0:
                            out_reg[i-1][1::2,:,-gap.shape[2]:] = roll_over[:,:,gap.shape[2]:]
                            print(f"\nnegative gap_roll_diff s > 0\n")


            else:
                roll_overs.append(None)

        elif s > 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,:shift_idx].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,:shift_idx] = replace_val

            
                if i > 0 and shifts[i-1] > 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                        prev_roll_over = roll_overs[i-2]

                        print(f"\nrollovers length: {len(roll_overs)}\n i: {i-2}\n")

                        gap_roll_diff = shift_idx - roll_over.shape[2]

                        print(f"\nshift_idx: {shift_idx}, roll_over shape[2]: {prev_roll_over.shape[2]}\n")

                        if roll_over_flag:
                            if gap_roll_diff >= 0:
                                odd_out[:,:,:prev_roll_over.shape[2]] = prev_roll_over
                                print(f"\npositive or neutral gap_roll_diff s > 0\n")

                            elif gap_roll_diff < 0:
                                odd_out[:,:,:shift_idx] = prev_roll_over[:,:,:shift_idx]
                                print(f"\nnegative gap_roll_diff s > 0\n")

            else:
                roll_overs.append(None)

        out[1::2,:,:] = odd_out

        out_reg.append(out)    

    output = np.concatenate(out_reg,axis=2)

    if fill_gaps:

        # find gaps
        gap_idxs = (output == replace_val)
        init_idxs = gap_idxs[1::2,:,:-1]
        final_idxs = gap_idxs[1::2,:,1:]

        # optional kwargs for viewer.add_* method
        add_kwargs2 = {"name": f"{name}_ascan_subpix_reg_debug"}

        # optional layer type argument
        layer_type2 = "labels"

        debug = np.empty_like(output)
        debug[gap_idxs] = 1
        debug = debug.astype('uint8')
        
        debug_label = Layer.create(debug,add_kwargs2,layer_type2)
        yield debug_label

        gap_starts = (init_idxs < final_idxs)
        gap_ends = (init_idxs > final_idxs)

        gap_starts = gap_starts.nonzero()
        gap_ends = gap_ends.nonzero()

        print(f"\ngap starts: {gap_starts}\ngap ends: {gap_ends}\n")

        gap_start_idxs = np.unique(gap_starts[2])
        gap_end_idxs = np.unique(gap_ends[2])

        print(f"\ngap starts: {gap_start_idxs}\ngap ends: {gap_end_idxs}\n")


        if len(gap_start_idxs) < len(gap_end_idxs):
            loops = len(gap_end_idxs)
        else:
            loops = len(gap_start_idxs)

        for i in tqdm(range(loops),desc="Filling middle gaps\n"):

            if len(gap_start_idxs) < len(gap_end_idxs):
                if i == 0:
                    s_idx = 0
                    e_idx = gap_end_idxs[i] + 1
                    num_tile = e_idx
                    vals = np.tile(output[1::2,:,e_idx],(num_tile,1,1))
                    vals = np.transpose(vals,(1,2,0))
                    output[1::2,:,:e_idx+1] = vals
                    #print(f"\n\n\n\nCondition ONE!!!!!!!!!!!!!!!\n\n\n\n")
                else:
                    s_idx = gap_start_idxs[i-1]
                    e_idx = gap_end_idxs[i] + 1

                    print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")
                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1
                    print(f"\nnum: {num}\n")
                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
                    print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")
                    output[1::2,:,s_idx:e_idx+1] = interp_out
                    #print(f"\n\n\n\nCondition TWO!!!!!!!!!!!!!!!\n\n\n\n")

            elif  len(gap_start_idxs) >= len(gap_end_idxs):
            
                if i < len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = gap_end_idxs[i] + 1

                    print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")
                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1
                    print(f"\nnum: {num}\n")
                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
                    print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")
                    output[1::2,:,s_idx:e_idx+1] = interp_out
                elif i >= len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = output.shape[2] - 1
                    num_tile = e_idx-(s_idx)
                    vals = np.tile(output[1::2,:,s_idx],(num_tile,1,1))
                    vals = np.transpose(vals,(1,2,0))

                    output[1::2,:,s_idx+1:] = vals #output[1::2,:,s_idx]
                    pass
                else:
                    print(f"\nWhy are we here near line 950?\n")

    layer = Layer.create(output,add_kwargs,layer_type)

    show_info(f'A-scan registration thread has completed')
    yield layer

def a_scan_reg_subpix_gen(vol:Image, settings:Dict, sub_pixel_threshold:float=0.5, fill_gaps=True,roll_over_flag=True,debug=False) -> List[Layer]:
    ''''''

    #from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    show_info(f'A-scan registration thread has started')

    #sections = 4
    data = vol.data
    name = vol.name

    replace_val = round(data.max()+2)

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_ascan_subpix_reg"}

    # optional layer type argument
    layer_type = "image"

    regions = settings["regions"]
    shifts = settings["shifts"]

    out_reg = []
    roll_overs = []

    for i,s in tqdm(enumerate(shifts),desc="Shifting Regions\n"):
        shift2 = round(s)
        shift_idx = abs(shift2)

        out = np.empty_like(regions[i])
        out[::2,:,:] = regions[i][::2,:,:]

        input_ = np.fft.fft2(regions[i][1::2,:,:])
        result = fourier_shift(input_,(0.0,0.0,s),axis=2)
        result = np.fft.ifft2(result)
        odd_out = result.real

        if s < 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,-shift_idx:].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,-shift_idx:] = replace_val

                
                if i > 0 and shifts[i-1] < 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                    out_reg_odd = out_reg[i-1][1::2,:,:]

                    axis_0_len,axis_1_len = out_reg_odd.shape[0], out_reg_odd.shape[1]

                    print(f"\nout_reg_odd shape: {out_reg_odd.shape}\n")

                    print(f"\nout_reg_odd[{i-1}] range: ({out_reg_odd.min()}, {out_reg_odd.max()})\n")

                    gap_idx = (out_reg_odd == replace_val)

                    gap = out_reg_odd[gap_idx]

                    gap.shape = (axis_0_len,axis_1_len,-1)

                    print(f"\nout_reg gap shape: {gap.shape}\nroll_over shape: {roll_over.shape}\n")

                    gap_roll_diff = gap.shape[2] - roll_over.shape[2]

                    if roll_over_flag:
                        if gap_roll_diff >= 0:
                            out_reg[i-1][1::2,:,-roll_over.shape[2]:] = roll_over
                            print(f"\npositive or neutral gap_roll_diff s < 0\n")
                        elif gap_roll_diff < 0:
                            out_reg[i-1][1::2,:,-gap.shape[2]:] = roll_over[:,:,gap.shape[2]:]
                            print(f"\nnegative gap_roll_diff s > 0\n")


            else:
                roll_overs.append(None)

        elif s > 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,:shift_idx].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,:shift_idx] = replace_val

            
                if i > 0 and shifts[i-1] > 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                        prev_roll_over = roll_overs[i-2]

                        print(f"\nrollovers length: {len(roll_overs)}\n i: {i-2}\n")

                        gap_roll_diff = shift_idx - roll_over.shape[2]

                        print(f"\nshift_idx: {shift_idx}, roll_over shape[2]: {prev_roll_over.shape[2]}\n")

                        if roll_over_flag:
                            if gap_roll_diff >= 0:
                                odd_out[:,:,:prev_roll_over.shape[2]] = prev_roll_over
                                print(f"\npositive or neutral gap_roll_diff s > 0\n")

                            elif gap_roll_diff < 0:
                                odd_out[:,:,:shift_idx] = prev_roll_over[:,:,:shift_idx]
                                print(f"\nnegative gap_roll_diff s > 0\n")

            else:
                roll_overs.append(None)

        out[1::2,:,:] = odd_out

        out_reg.append(out)    

    output = np.concatenate(out_reg,axis=2)

    if fill_gaps:

        # find gaps
        gap_idxs = (output == replace_val)
        init_idxs = gap_idxs[1::2,:,:-1]
        final_idxs = gap_idxs[1::2,:,1:]

        # debug

        # optional kwargs for viewer.add_* method
        add_kwargs2 = {"name": f"{name}_ascan_subpix_reg_debug"}

        # optional layer type argument
        layer_type2 = "labels"

        if debug:
            debug = np.empty_like(output)
            debug[gap_idxs] = 1
            debug = debug.astype('uint8')
            
            debug_label = Layer.create(debug,add_kwargs2,layer_type2)
            yield debug_label

        gap_starts = (init_idxs < final_idxs)
        gap_ends = (init_idxs > final_idxs)

        gap_starts = gap_starts.nonzero()
        gap_ends = gap_ends.nonzero()

        print(f"\ngap starts: {gap_starts}\ngap ends: {gap_ends}\n")

        gap_start_idxs = np.unique(gap_starts[2])
        gap_end_idxs = np.unique(gap_ends[2])

        print(f"\ngap starts: {gap_start_idxs}\ngap ends: {gap_end_idxs}\n")


        if len(gap_start_idxs) < len(gap_end_idxs):
            loops = len(gap_end_idxs)
        else:
            loops = len(gap_start_idxs)

        for i in tqdm(range(loops),desc="Filling middle gaps\n"):

            if len(gap_start_idxs) < len(gap_end_idxs):
                if i == 0:
                    s_idx = 0
                    e_idx = gap_end_idxs[i] + 1
                    num_tile = e_idx
                    vals = np.tile(output[1::2,:,e_idx],(num_tile,1,1))
                    vals = np.transpose(vals,(1,2,0))
                    output[1::2,:,:e_idx+1] = vals

                else:
                    s_idx = gap_start_idxs[i-1]
                    e_idx = gap_end_idxs[i] + 1

                    print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")
                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1
                    print(f"\nnum: {num}\n")
                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
                    print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")
                    output[1::2,:,s_idx:e_idx+1] = interp_out

            elif  len(gap_start_idxs) >= len(gap_end_idxs):
            
                if i < len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = gap_end_idxs[i] + 1

                    print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")
                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1
                    print(f"\nnum: {num}\n")
                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
                    print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")
                    output[1::2,:,s_idx:e_idx+1] = interp_out
                elif i >= len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = output.shape[2] - 1
                    num_tile = e_idx-(s_idx)
                    vals = np.tile(output[1::2,:,s_idx],(num_tile,1,1))
                    vals = np.transpose(vals,(1,2,0))
                    output[1::2,:,s_idx+1:] = vals #output[1::2,:,s_idx]
                    pass
                else:
                    print(f"\nWhy are we here near line 950?\n")

    layer = Layer.create(output,add_kwargs,layer_type)

    show_info(f'A-scan registration thread has completed')

    yield layer