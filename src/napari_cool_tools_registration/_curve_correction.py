from fileinput import filename
from networkx import is_empty
import numpy as np
from napari.utils import progress
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QLabel,
    QFileDialog,
)

from napari.qt.threading import create_worker
from qtpy import QtCore
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
import napari
from napari.utils import progress
from scipy.interpolate import interp1d
import cupy as cu
from cupyx.scipy.ndimage import map_coordinates
# from scipy.ndimage import map_coordinates
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import shift
import scipy.ndimage
from cupyx.scipy.interpolate import interpn
import time
import scipy.io as sio
import tifffile


#this is curve correction in 2D with cylindrical method
def curve_correction(image, pivot_point = 0, scan_angle = 140) -> Image:
    def curve_correction_2D(image,coordinates_gpu,resolution):

        #send data to GPU
        data_gpu = cu.asarray(image)

        #map coordinates in gpu
        data_gpu = map_coordinates(data_gpu, coordinates_gpu)

        #retrieve data from gpu
        new_image = cu.asnumpy(data_gpu)
        new_image = new_image.reshape((resolution,resolution))
        # new_image = new_image[int(resolution/2):,:]
        new_image = new_image.astype(np.float32)

        return new_image

    def dewrap_2D(data, angle = 140):
        h = data.shape[0]
        w = data.shape[1]

        Xn = np.arange(w)
        x_org = w*np.sin((0.5*np.pi/w)*Xn)

        vol_out = np.empty_like(data)

        for j in range(h):
            f = interp1d(Xn,data[j,:], bounds_error=False, fill_value=0.0)
            vol_out[j,:] = f(x_org)

        return vol_out


    show_info("Curve Correction in Progress.")
    data = image.data
    name = f"{image.name}_curve_corrected"

    data = data.transpose((0, 2, 1))

    output_size = np.max(data[:,:,0].shape)
    scan_range = output_size#the biggest width

    r = np.linspace(0, scan_range, output_size)
    th = np.linspace(0, np.pi*2, output_size) #did 360 degree scan at the center of image

    R, TH = np.meshgrid(r,th)

    x = R*np.cos(TH) # put it in the center of the original image
    x = x + output_size*0.5 - 0.5
    y = R*np.sin(TH) # we need to consider the unequally spaced pixel
    y = y + output_size*0.5 - 0.5

    output_image = np.zeros((output_size,int(output_size/2),data.shape[2]))

    for fnum in range(0,data.shape[2]):
        image = data[:,:,fnum]
        image = resize(image,(output_size,output_size))
        image_gpu = cu.asarray(image)

        coordinates = np.array([x, y])
        coordinates_gpu = cu.asarray(coordinates)

        new_image = map_coordinates(image_gpu, coordinates_gpu)
        new_image = cu.asnumpy(new_image)

        new_image[:,int(output_size/2):] = 0

        #dewrap image to 70 degress instead of 90
        wrapped_image = new_image[:,:int(output_size/2)]
        wrapped_image = dewrap_2D(wrapped_image,scan_angle)
        new_image[:,:int(output_size/2)] = wrapped_image

        output_image[:,:,fnum] = new_image[:,:int(output_size/2)]
        yield 1
    
    output_image = output_image.transpose((0, 2, 1))
    data = output_image

    #################
    #curve correction 2D
    # pivot_point is in pixel
    padding = int(np.round(pivot_point))

    angle = 180 - scan_angle
    angle = angle*0.5

    ###bend in x direction
    radius = data.shape[1] + padding
    resolution = radius*2

    x = np.linspace(0, radius, resolution)
    y = np.linspace(0, radius, resolution) 

    #this is the target coordinates
    X, Y = np.meshgrid(x,y)

    X = X - radius*0.5
    Y = Y - radius*0.5

    num_r = data.shape[1]
    num_theta = data.shape[2]

    # This is location in the original image
    r = np.linspace(0, num_r, num_r*2)
    th = np.linspace(90/180*np.pi, angle/180*np.pi, num_theta)

    w = num_theta
    Xn = np.arange(w)
    x_org = w*np.sin((scan_angle*0.5/180)*np.pi*(1.0/w*Xn))
    f = interp1d(Xn,th, bounds_error=False, fill_value=0.0)
    th = f(x_org)

    #this is the new target location
    new_r = np.sqrt(X*X+Y*Y)
    new_th = np.arctan2(Y, X)
    new_th[np.isnan(new_th)] = 0

    ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value="extrapolate")
    ith = interp1d(th, np.arange(len(th)), bounds_error=False, fill_value="extrapolate")

    new_ir = ir(new_r.ravel())
    new_ith = ith(new_th.ravel())

    output_image = np.zeros((data.shape[0], int(resolution/2), int(resolution/2)), dtype = np.float32)

    coordinates = np.array([new_ir, new_ith])
    coordinates_gpu = cu.asarray(coordinates)

    for frame, image in enumerate(data):
        image = np.pad(image, ((padding, 0), (0, 0)), mode='constant', constant_values=0)
        image = curve_correction_2D(image, coordinates_gpu, resolution)
        output_image[frame] = image[int(resolution/2):, int(resolution/2):]
        yield 1

    output_image = output_image.transpose((1, 2, 0))
    data = output_image


#############################
    #put it back to cartesian
    radius = data.shape[0]
    resolution = radius*2

    x = np.linspace(0, radius, resolution)
    y = np.linspace(0, radius, resolution) 

    #this is the target coordinates
    X, Y = np.meshgrid(x,y)

    X = X - radius*0.5
    Y = Y - radius*0.5

    num_r = data.shape[1]
    num_theta = data.shape[2]

    # This is location in the original image
    r = np.linspace(0, num_r, num_r*2)
    th = np.linspace(0, np.pi*2, num_theta)

    #this is the new target location
    new_r = np.sqrt(X*X+Y*Y)
    new_th = np.arctan2(Y, X) + np.pi
    new_th[np.isnan(new_th)] = 0

    ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value="extrapolate")
    ith = interp1d(th, np.arange(len(th)), bounds_error=False, fill_value="extrapolate")

    new_ir = ir(new_r.ravel())
    new_ith = ith(new_th.ravel())

    output_image = np.zeros((data.shape[0],resolution, resolution), dtype = np.float32)

    coordinates = np.array([new_ir, new_ith])
    coordinates_gpu = cu.asarray(coordinates)

    for frame, image in enumerate(data):
        image = curve_correction_2D(image, coordinates_gpu, resolution)
        output_image[frame] = image
        yield 1

    output_image = output_image[:,int(resolution/8):resolution-int(resolution/8),int(resolution/8):resolution-int(resolution/8)]
    output_image = output_image.transpose((1, 0, 2))


    add_kwargs = {"name":name}
    layer_type = "image"
    new_layer = Layer.create(output_image,add_kwargs,layer_type)

    show_info("Curve Correction is Finished.")

    return new_layer



# #this is curve correction in 2D old which is wrong
# def curve_correction(image, pivot_point = 0, scan_angle = 140) -> Image:

#     def dewrap_2D(data, angle = 140):
#         h = data.shape[0]
#         w = data.shape[1]

#         Xn = np.arange(w)
#         x_org = 0.5*w*np.sin(np.pi*(1.0/w*Xn - 0.5)) + 0.5*w

#         vol_out = np.empty_like(data)

#         for j in range(h):
#             f = interp1d(Xn,data[j,:], bounds_error=False, fill_value=0.0)
#             vol_out[j,:] = f(x_org)

#         return vol_out

#     def curve_correction_2D(image,coordinates_gpu,resolution):

#         #send data to GPU
#         data_gpu = cu.asarray(image)

#         #map coordinates in gpu
#         data_gpu = map_coordinates(data_gpu, coordinates_gpu)

#         #retrieve data from gpu
#         new_image = cu.asnumpy(data_gpu)
#         new_image = new_image.reshape((resolution,resolution))
#         #new_image = new_image[int(resolution/2):,:]
#         new_image = new_image.astype(np.float32)

#         return new_image

#     show_info("Curve Correction in Progress.")
#     data = image.data
#     name = f"{image.name}_curve_corrected"

#     # pivot_point is in pixel
#     padding = int(np.round(pivot_point))

#     start_angle = 180 - scan_angle
#     start_angle = start_angle*0.5 #20
#     end_angle = start_angle + scan_angle #160 = 140 + 20 
#     start_angle = start_angle*np.pi/180    
#     end_angle = end_angle*np.pi/180

#     ###bend in x direction
#     radius = data.shape[1] + padding
#     resolution = radius*2

#     x = np.linspace(0, radius, resolution)
#     y = np.linspace(0, radius, resolution) 

#     #this is the target coordinates
#     X, Y = np.meshgrid(x,y)

#     X = X - radius*0.5
#     Y = Y - radius*0.5

#     num_r = data.shape[1]
#     num_theta = data.shape[2]

#     # This is location in the original image
#     r = np.linspace(0, num_r, num_r*2)
#     th = np.linspace(np.pi + start_angle,np.pi + end_angle, num_theta)

#     w = num_theta
#     Xn = np.arange(w)
#     x_org = 0.5*w*np.sin((scan_angle/180)*np.pi*(1.0/w*Xn - 0.5)) + 0.5*w
#     f = interp1d(Xn,th, bounds_error=False, fill_value=0.0)
#     th = f(x_org)

#     #this is the new target location
#     new_r = np.sqrt(X*X+Y*Y)
#     new_th = np.arctan2(Y, X) + np.pi
#     new_th[np.isnan(new_th)] = 0

#     ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value="extrapolate")
#     ith = interp1d(th, np.arange(len(th)), bounds_error=False, fill_value="extrapolate")

#     new_ir = ir(new_r.ravel())
#     new_ith = ith(new_th.ravel())

#     output_image = np.zeros((data.shape[0],int(resolution/2), resolution), dtype = np.float32)

#     coordinates = np.array([new_ir, new_ith])
#     coordinates_gpu = cu.asarray(coordinates)

#     for frame, image in enumerate(data):
#         image = dewrap_2D(image,scan_angle)
#         image = np.pad(image, ((padding, 0), (0, 0)), mode='constant', constant_values=0)
#         image = curve_correction_2D(image, coordinates_gpu, resolution)
#         output_image[frame] = image[int(resolution/2):,:]
#         yield 1

#     ###bend in y direction
#     data =  output_image.transpose(2,1,0)

#     radius = data.shape[1]
#     resolution = radius*2

#     x = np.linspace(0, radius, resolution)
#     y = np.linspace(0, radius, resolution) 

#     #this is the target coordinates
#     X, Y = np.meshgrid(x,y)

#     X = X - radius*0.5
#     Y = Y - radius*0.5

#     num_r = data.shape[1]
#     num_theta = data.shape[2]

#     # This is location in the original image
#     r = np.linspace(0, num_r, num_r*2)
#     th = np.linspace(np.pi + start_angle,np.pi + end_angle, num_theta)

#     w = num_theta
#     Xn = np.arange(w)
#     x_org = 0.5*w*np.sin((scan_angle/180)*np.pi*(1.0/w*Xn - 0.5)) + 0.5*w
#     f = interp1d(Xn,th, bounds_error=False, fill_value=0.0)
#     th = f(x_org)

#     #this is the new target location
#     new_r = np.sqrt(X*X+Y*Y)
#     new_th = np.arctan2(Y, X) + np.pi
#     new_th[np.isnan(new_th)] = 0

#     ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value="extrapolate")
#     ith = interp1d(th, np.arange(len(th)), bounds_error=False, fill_value="extrapolate")

#     new_ir = ir(new_r.ravel())
#     new_ith = ith(new_th.ravel())

#     output_image = np.zeros((data.shape[0],int(resolution/2), resolution), dtype = np.float32)

#     coordinates = np.array([new_ir, new_ith])
#     coordinates_gpu = cu.asarray(coordinates)

#     for frame, image in enumerate(data):
#         image = dewrap_2D(image,scan_angle)
#         image = curve_correction_2D(image, coordinates_gpu, resolution)
#         output_image[frame] = image[int(resolution/2):,:]
#         yield 1

#     add_kwargs = {"name":name}
#     layer_type = "image"
#     new_layer = Layer.create(output_image,add_kwargs,layer_type)

#     show_info("Curve Correction is Finished.")

#     return new_layer



def downsample_image(image,downsample_factor):
    show_info("Down Sampling in Progress.")
    data = image.data
    name = f"{image.name}_downsampled"

    output_image = scipy.ndimage.zoom(data, downsample_factor)

    add_kwargs = {"name":name}
    layer_type = "image"
    new_layer = Layer.create(output_image,add_kwargs,layer_type)

    show_info("Down Sampling is Finished.")

    return new_layer

def saveas_numpy(data,fname):
    show_info("Save as Numpy Started.")
    with open(fname, 'wb') as f:
        np.save(f, data)

    show_info("Save as Numpy Finished.")

def saveas_bigtiff(data,fname):
    show_info("Save as TIFF Started.")
    tifffile.imwrite(fname, data, bigtiff= True)


class Curve_Correction_Widget(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Correction Panel")

        #this spin box is used to change the axial resolution
        self.axial_resolution_label = QLabel("Imaging Range (mm)")
        self.layout().addWidget(self.axial_resolution_label)
        self.axial_resolution = QDoubleSpinBox()
        self.axial_resolution.setSingleStep(1.0)
        self.axial_resolution.setDecimals(2)
        self.axial_resolution.setMinimum(-100000.00)
        self.axial_resolution.setMaximum(100000.00)
        self.axial_resolution.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.axial_resolution.setValue(10.0)
        self.layout().addWidget(self.axial_resolution)

        #this spin box is used to change the pivot point location
        self.pivot_point_label = QLabel("Pivot Point (mm)")
        self.layout().addWidget(self.pivot_point_label)
        self.pivot_point = QDoubleSpinBox()
        self.pivot_point.setSingleStep(0.1)
        self.pivot_point.setDecimals(2)
        self.pivot_point.setMinimum(0.00)
        self.pivot_point.setMaximum(1000.00)
        self.pivot_point.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pivot_point.setValue(1.0)
        self.layout().addWidget(self.pivot_point)

        #this spin box is used to change scan angle of the OCT
        self.degree_label = QLabel("Scan Angle (<sup>o</sup>)")
        self.layout().addWidget(self.degree_label)
        self.scan_angle = QDoubleSpinBox()
        self.scan_angle.setSingleStep(0.1)
        self.scan_angle.setDecimals(2)
        self.scan_angle.setMinimum(0.00)
        self.scan_angle.setMaximum(360.00)
        self.scan_angle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scan_angle.setValue(140.0)
        self.layout().addWidget(self.scan_angle)

        self.curve_button = QPushButton("Correct Curve")
        self.layout().addWidget(self.curve_button)
        self.curve_button.clicked.connect(self.on_curve_button_clicked)

        #this is just a dummy function to initialize a worker thread
        dummy_function = lambda : 10
        self.worker = create_worker(dummy_function)

        #this spin box is used to change the down sample
        self.downsample_label = QLabel("Downsample Factor")
        self.layout().addWidget(self.downsample_label)
        self.downsample_factor = QDoubleSpinBox()
        self.downsample_factor.setSingleStep(0.1)
        self.downsample_factor.setDecimals(2)
        self.downsample_factor.setMinimum(0.00)
        self.downsample_factor.setMaximum(1.00)
        self.downsample_factor.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.downsample_factor.setValue(0.5)
        self.layout().addWidget(self.downsample_factor)

        self.downsample_button = QPushButton("Downsample")
        self.layout().addWidget(self.downsample_button)
        self.downsample_button.clicked.connect(self.on_downsample_button_clicked)

        self.savenumpy_button = QPushButton("Save as numpy")
        self.layout().addWidget(self.savenumpy_button)
        self.savenumpy_button.clicked.connect(self.on_savenumpy_button_clicked)

        self.savetiff_button = QPushButton("Save as tiff")
        self.layout().addWidget(self.savetiff_button)
        self.savetiff_button.clicked.connect(self.on_savetiff_button_clicked)


    def on_savetiff_button_clicked(self):
        if self.worker.is_running:
            show_info("A Curve Correction process is running. Please Wait!")
            return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a layers shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        name = f"{current_layer.name}.tif"

        fileName, filters = QFileDialog.getSaveFileName(self, "Save File", name, "TIFF File (*.tif)")
            
        progress_bar = progress()
        progress_bar.set_description("Save as TIFF")
        progress_bar.display()

        data = current_layer.data

        self.worker = create_worker(saveas_bigtiff, data, fileName)
        
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()

    def on_savenumpy_button_clicked(self):
        if self.worker.is_running:
            show_info("A Curve Correction process is running. Please Wait!")
            return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a layers shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        name = f"{current_layer.name}.npy"

        fileName, filters = QFileDialog.getSaveFileName(self, "Save File", name, "Numpy Array (*.npy)")
            
        progress_bar = progress()
        progress_bar.set_description("Save as Numpy")
        progress_bar.display()

        data = current_layer.data

        self.worker = create_worker(saveas_numpy, data, fileName)
        
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()

    def on_downsample_button_clicked(self):
        if self.worker.is_running:
             show_info("A Curve Correction process is running. Please Wait!")
             return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a layers shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        progress_bar = progress()
        progress_bar.set_description("Down Sampling Image(s)")
        progress_bar.display()

        self.worker = create_worker(downsample_image, current_layer, self.downsample_factor.value())
        
        self.worker.returned.connect(self.viewer.add_layer)
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()



    def on_curve_button_clicked(self):
        if self.worker.is_running:
             show_info("A Curve Correction process is running. Please Wait!")
             return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a line shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return

        scan_angle = self.scan_angle.value()

        # calculate pixel spacing from imaging range
        pixel_spacing = self.axial_resolution.value()*1000/current_layer.data.shape[1]

        print(f"Pixel Spacing = ",pixel_spacing)

        # calculate pivot in pixel
        pivot_point = self.pivot_point.value()*1000/pixel_spacing
        print(f"Pivot_point = ",pivot_point)

        total = current_layer.data.shape[0] +  (current_layer.data.shape[1] + pivot_point)*2
        progress_bar = progress(total=int(np.ceil(total)))
        progress_bar.set_description("Correcting Image Curvature")

        self.worker = create_worker(curve_correction, current_layer, pivot_point = pivot_point
                                    , scan_angle = scan_angle)
        
        self.worker.returned.connect(self.viewer.add_layer)
        self.worker.yielded.connect(progress_bar.update)
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()
