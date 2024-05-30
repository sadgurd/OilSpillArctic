#DataCollection
from __future__ import annotations
import math
from IPython.display import clear_output, display
import requests
import getpass
from sentinelhub import SHConfig
import datetime
import os
import matplotlib.pyplot as plt
from typing import Any
from pandas import to_datetime, date_range
import matplotlib.animation as animation
from PIL import Image
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import Sequence, Tuple, List, Optional
import cv2
from empatches import BatchPatching
from torchvision import transforms
import torch
import files.config as Config
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.dates import DateLocator, DateFormatter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from imutils import paths

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    bbox_to_resolution,
    transform_point,
    get_utm_crs,
    aws,
    SentinelHubCatalog,
    Geometry,
)
##########################
#########################
#########################
#########################

# https://youtu.be/HrGn4uFrMOM

"""

Original code is from the following source. It comes with MIT License so please mention
the original reference when sharing.

The original code has been modified to fix a couple of bugs and chunks of code
unnecessary for smooth tiling are removed. 

# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE

"""
"""Perform smooth predictions on an image from tiled prediction patches."""


import scipy.signal
from tqdm import tqdm

import gc

def lee_filter(img, size):
    #https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:

    PLOT_PROGRESS = False

def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)      #SREENI: Changed from 3, 3, to 1, 1 
        wind = wind * wind.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind

def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(ret)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return ret

def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    # gc.collect()
    return ret

def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs

def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)

def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):            #SREENI: Changed padx to pady (Bug in original code)
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()

    subdivs = pred_func(torch.FloatTensor(subdivs/255).permute(0, 3, 1, 2).to(Config.device)).detach().cpu().numpy()
    subdivs = subdivs.transpose(0, 2, 3, 1)
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()

    return subdivs

def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):                #SREENI: Changed padx to pady (Bug in original code)
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    """if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()"""
    return prd

#########################################################
#########################################################
#########################################################
#########################################################
evalscripts = {"evalscript1":  """
    //VERSION=3
    function setup() {
        return {
            input: ["HH", "dataMask"],
            output: { bands: 4 }
        }
    }

    function evaluatePixel(sample) {
        var value = 2 * sample.HH;
        value = Math.min(5, Math.max(-30, value));
        return [value, value, value, sample.dataMask];
    }
    """,

"evalscript2": """//VERSION=3
function setup() {
  return {
    input: ["HH", "dataMask"],
    output: [
      { id: "default", bands: 4 },
      { id: "eobrowserStats", bands: 1 },
      { id: "dataMask", bands: 1 },
    ],
  }
}

function evaluatePixel(samples) {
  const value = Math.max(0, Math.log(samples.HH) * 0.21714724095 + 1);
  return {
    default: [value, value, value, samples.dataMask],
    eobrowserStats: [(10 * Math.log(samples.HH)) / Math.LN10],
    dataMask: [samples.dataMask],
  };
}

// ---
/*
// displays HH in decibels from -20 to 0
// the following is simplified below
// var log = 10 * Math.log(HH) / Math.LN10;
// var val = Math.max(0, (log + 20) / 20);

return [Math.max(0, Math.log(HH) * 0.21714724095 + 1)];
*/""",

"evalscript3": """//VERSION=3
function setup() {
  return {
    input: ["VV", "dataMask"],
    output: [
      { id: "default", bands: 4 },
      { id: "eobrowserStats", bands: 1 },
      { id: "dataMask", bands: 1 },
    ],
  };
}

function evaluatePixel(samples) {
  const value = Math.max(0, Math.log(samples.VV) * 0.21714724095 + 1);
  return {
    default: [value, value, value, samples.dataMask],
    eobrowserStats: [(10 * Math.log(samples.VV)) / Math.LN10],
    dataMask: [samples.dataMask],
  };
}

// ---
/*
  // displays VV in decibels from -20 to 0
  // the following is simplified below
  // var log = 10 * Math.log(VV) / Math.LN10;
  // var val = Math.max(0, (log + 20) / 20);
  
  return [Math.max(0, Math.log(VV) * 0.21714724095 + 1)];
*/""",
"evalscript4": """//VERSION=3
function setup() {
  return {
    input: ["HH", "dataMask"],
    output: [
      { id: "default", bands: 4 },
      { id: "eobrowserStats", bands: 1 },
      { id: "dataMask", bands: 1 },
    ],
  }
}

function evaluatePixel(samples) {
  return {
    default: [2 * samples.HH, 2 * samples.HH, 2 * samples.HH, samples.dataMask],
    eobrowserStats: [samples.HH],
    dataMask: [samples.dataMask],
  };
}"""
}

def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]

def setup_config(profile: str,
                 client: str,
                 secret: str,
                 token_url: Optional[str] = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                 base_url: Optional[str] = "https://sh.dataspace.copernicus.eu"):
    
    """Setup of sentinel client credentials (for the given Copernicus hub is used)
    Following links are used:
    https://sentinelhub-py.readthedocs.io/en/latest/configure.html                  //Configuration
    https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html   //Data request setup
    https://sentinelhub-py.readthedocs.io/en/latest/examples/data_collections.html  //Setup
    https://dataspace.copernicus.eu/news/2023-9-28-accessing-sentinel-mission-data-new-copernicus-data-space-ecosystem-apis
    https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.geometry.html //Geometry
    """

    #First we need to setup the configuration for the sentinelhub
    config = SHConfig()                                                          #Configuration Class
    #config_location = SHConfig.get_config_location()                             #Location of the configuration file
    config.instance_id = profile                                                 #Profile id
    config.sh_client_id = client                                                #ID for OAuth from sentinelhub
    config.sh_client_secret = secret                                            #Secret for OAuth from sentinelhub
    config.sh_token_url = token_url
    config.sh_base_url = base_url
    config.save(profile)                                                         #Saves data to the config file

    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! To use process API, please provide the credentials (OAuth client ID and client Secret)")

def Connect(profile: str):
    config = SHConfig(profile)
    for collection in DataCollection.get_available_collections():
        collection.service_url = config.sh_base_url
    return config

#Plotting function
def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def zoom(img, x, y, zoom):
    h, w, _ = img.shape
    print(w, h)
    L, H  = w/zoom, h /zoom
    x1, x2, y1, y2 = np.floor(abs(x - L/2)),np.floor(abs(x + L/2)), np.ceil(abs(y - H/2)), np.ceil(abs(y + H/2))
    print(x1,x2,y1,y2)
    image = img[int(y1):int(y2), int(x1):int(x2)]
    image = cv2.resize(image, (w, h))
    return image
#Functions for collecting the data
def collect_data(time_interval, AOI, Size, config, evalscript, datacollection):
    request = SentinelHubRequest(evalscript = evalscript,
                                input_data = [
                                    SentinelHubRequest.input_data(
                                        data_collection=datacollection,
                                        time_interval=time_interval,
                                    )
                                ],
                                responses = [SentinelHubRequest.output_response("default", MimeType.JPG )],
                                                                                bbox=AOI,
                                                                                size=Size,
                                                                                config=config,
                                                                                    )
    image = request.get_data()

    return image

#Function to sort out the data that isn't complete or overlapping dates of the loaded polygons
def extract_possible_dates(time_interval, coords, config, datacollection):
    catalog = SentinelHubCatalog(config=config)
    aoi_bbox = BBox(bbox=coords, crs=CRS.WGS84)
    search_iterator = catalog.search(
            datacollection,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["geometry.coordinates", "properties.datetime"], "exclude": []},
        )

    results = list(search_iterator)
    print("Total number of results:", len(results))
    dates = [pd.to_datetime(result["properties"]["datetime"]).date().strftime("%Y-%m-%d") for result in results]
    print(dates)
    coordinates = [result['geometry']["coordinates"][0] for result in results ]
    collected_coords = []
    for d in dict.fromkeys(dates):
            collected_coords.append([coordinates[dates.index(d)+c] for c in range(dates.count(d))])
    multipolyray = []
    for i in range(len(collected_coords)):
        temp_poly_array = []
        for l in range(len(collected_coords[i][:])):
            multicoordray = []
            for s in collected_coords[i][l]:
                multicoordray.append((s[0],s[1]))
            temp_poly_array.append(Polygon(multicoordray))
        if len(temp_poly_array) != 1: 
            multipolyray.append(unary_union(temp_poly_array))
        else:
            multipolyray.append(temp_poly_array[0])

    location = Polygon(((coords[0],coords[1]),
                        (coords[2],coords[1]),
                        (coords[2],coords[3]),
                        (coords[0],coords[1]),
                        (coords[0],coords[1])))

    is_in = gpd.GeoSeries(multipolyray).contains(location)
    i = 0
    possible_dates = dict.fromkeys(dates)

    for d in possible_dates:
        print(f'date {d} is in {is_in[i]}')
        i +=1 
    
    dates = [date for date, is_in_value in zip(possible_dates, is_in) if is_in_value]
    print(dates, "\n")
    return dates

#Function to extract the dates with overlapping polygons
def extract_dates(time_interval, coords, config, datacollection):
    catalog = SentinelHubCatalog(config=config)
    aoi_bbox = BBox(bbox=coords, crs=CRS.WGS84)
    search_iterator = catalog.search(
            datacollection,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["geometry.coordinates", "properties.datetime"], "exclude": []},
        )

    results = list(search_iterator)
    print("Total number of results:", len(results))
    dates = [pd.to_datetime(result["properties"]["datetime"]).date().strftime("%Y-%m-%d") for result in results]
    
    return dates

#Function to calculate the parameters for the data collection
def calculate_parameters(BBCoords, ImWidth, ImHeight, resolution, time_interval, datacollection, config, filtered):
    CoordBBox = BBox(bbox=BBCoords, crs=CRS.WGS84)                     #Setup bbox, crs is a kind of coordinate reference system          
    BBoxSize = bbox_to_dimensions(CoordBBox, resolution = resolution)  #Size of the bbox    
    if filtered == True:
        dates = list(reversed(dict.fromkeys(extract_possible_dates(
            time_interval, BBCoords, config, datacollection))))           #Finds the amount of dates available and fully complete
    else:
        dates = list(reversed(dict.fromkeys(extract_dates(
            time_interval, BBCoords, config, datacollection))))            #Finds the amount of dates available that arent fully complete
    
    total_possible_days = len(dates)                                    #Total amount of days available

    CoordsUtm1 = transform_point((BBCoords[0], BBCoords[1]),source_crs = CRS.WGS84,
                             target_crs = get_utm_crs(BBCoords[0], BBCoords[1]))
    CoordsUtm2 = transform_point((BBCoords[2], BBCoords[3]),source_crs = CRS.WGS84
                                ,target_crs = get_utm_crs(BBCoords[2], BBCoords[3]))

    UTMBBox = BBox(bbox=(CoordsUtm1[0], CoordsUtm1[1], CoordsUtm2[0], CoordsUtm2[1]), crs = get_utm_crs(BBCoords[0], BBCoords[1]))
    utm_BBoxSize = bbox_to_dimensions(UTMBBox, resolution = resolution)

    Distance_utmx, Distance_utmy = abs(CoordsUtm1[0] - CoordsUtm2[0]), abs(CoordsUtm1[1] - CoordsUtm2[1])

    partition_size_x = Distance_utmx * ((ImWidth/utm_BBoxSize[0]))   
    partition_size_y = Distance_utmy * ((ImHeight/utm_BBoxSize[1]))  #Lat / Long coordinates difference and resolution scale.

    BBoxPartition = UTMBBox.get_partition(size_x=partition_size_x,
                                            size_y=partition_size_y)  #Partition of the bbox for the data collection

    for i in range(len(BBoxPartition)):
        for j in range(len(BBoxPartition[0][:])):
            BBoxPartition[i][j] = BBox.transform(BBoxPartition[i][j], crs = CRS.WGS84)
    
    res_up = bbox_to_resolution(BBoxPartition[0][0], width=ImWidth, height=ImHeight)     #Resolution update to end up with 1250x650 image
    reBBoxSize = bbox_to_dimensions(BBoxPartition[0][0], resolution = res_up)              #Size of the bbox at the updated resolution

    ReBBCoords = (BBoxPartition[-1][0].lower_left[0],
            BBoxPartition[-1][0].lower_left[1],
            BBoxPartition[0][-1].upper_right[0],
            BBoxPartition[0][-1].upper_right[1])
    CoordReBBox = BBox(bbox=ReBBCoords, crs=CRS.WGS84)                     #Setup bbox, crs is a kind of coordinate reference system        
    ogReBBoxSize = bbox_to_dimensions(CoordReBBox, resolution = res_up)  #Size of the bbox at the updated resolution

    print(f"\nImage shape at {resolution} m resolution: {BBoxSize} pixels\n")
    print(f'\nResized image shape at {res_up} m resolution: {ogReBBoxSize} pixels\n')
    print(f"\nResized image shape at ")
    print(f'Partition size: {reBBoxSize} pixels, with resolution {res_up} m and shape {len(BBoxPartition)} x {len(BBoxPartition[0][:])} partitions.\n')
    #print(f'resolution error percentage: in x {round(abs(res_up[0]/resolution*100-100),2)}%, in y: {(round(abs(res_up[1]/resolution*100-100),2))} %\n')
    print(f'Original bbox coordinates: {BBCoords}, Partioned BBox coordinates: {BBoxPartition[0][0].upper_right[0],BBoxPartition[0][0].upper_right[1], BBoxPartition[-1][-1].lower_left[0],BBoxPartition[-1][-1].lower_left[1]}\n')
    return dates, res_up, total_possible_days, BBoxPartition, reBBoxSize

def median_filter(img):
    #https://likegeeks.com/median-filter-numpy-python/ - Code taken from here
    kernel_size = 7
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)

    half_kernel_size = kernel_size // 2
    padded_image = np.pad(img, pad_width=half_kernel_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(padded_image)
    print(padded_image[1].shape, padded_image[1].shape, padded_image[:,1].shape)
    for i in range(half_kernel_size, padded_image.shape[0] - half_kernel_size):
        for j in range(half_kernel_size, padded_image.shape[1] - half_kernel_size):
            window = padded_image[i - half_kernel_size:i + half_kernel_size + 1, j - half_kernel_size:j + half_kernel_size + 1]
            filtered_image[i, j] = np.median(window)

    filtered_image = filtered_image[half_kernel_size:-half_kernel_size, half_kernel_size:-half_kernel_size]
    
    print(filtered_image.shape, half_kernel_size, padded_image.shape, img.shape)
    return filtered_image

def get_frames(BBoxPartition, reBBoxSize, local_time_interval, config, evalscript, datacollection: Optional[DataCollection] = DataCollection.SENTINEL1, bordered: Optional[bool] = False):
    bordered_frames = []

    N = len(BBoxPartition)
    M = len(BBoxPartition[0][:])

    for i in range(N):
        for j in range(M):
            frame = collect_data(local_time_interval, BBoxPartition[i][j],
                                 reBBoxSize, config, evalscript, datacollection)
            if bordered == True:
                bordered_frame = np.pad(frame[-1], ((5,5),(5,5), (0,0)), mode='constant')
                bordered_frames.append(bordered_frame)
            else:
                bordered_frames.append(frame[-1])
            
    return bordered_frames

def visualize_patches(BBCoords,
                      time_interval,
                      mydpi: int,
                      config,
                      resolution,
                      size: Optional[Tuple[int, int]] = (320, 320),
                      datacollection: Optional[DataCollection] = DataCollection.SENTINEL1,
                      evalscript: Optional[str] = evalscripts["evalscript1"]
        ):
    """
    
    Function for visualizing the patched area, and collect the data from copernicus hub. If multiple dates have been chosen, and visualize_days = False
    then only the first date will be visualized.

    """
    dates, res_up, total_possible_days, BBoxPartition, reBBoxSize = calculate_parameters(BBCoords, size[1], size[0], resolution, time_interval, datacollection, config, filtered = False)
    
    fig, ax = plt.subplots(nrows = 1, ncols=2, dpi=mydpi, frameon=False) #figsize=size/mydpi
    ax[0].axis('off')
    ax[1].axis('off')
    
    N = len(BBoxPartition)
    M = len(BBoxPartition[0][:])

    local_time_interval = dates[0], dates[0]
    CoordBBox = BBox(bbox=BBCoords, crs=CRS.WGS84)                     #Setup bbox, crs is a kind of coordinate reference system        
    ogBBoxSize = bbox_to_dimensions(CoordBBox, resolution = resolution)

    if ogBBoxSize[0] > 2000 or ogBBoxSize[1] > 2000:
        ogBBoxSize = bbox_to_dimensions(CoordBBox, resolution = (resolution[0]*10, resolution[1]*10))
        if ogBBoxSize[0] > 2000 or ogBBoxSize[1] > 2000:
            print("BBOX AREA TOO LARGE")
            return

    og_image = collect_data(local_time_interval, CoordBBox, ogBBoxSize, config, evalscript, datacollection)
    
    
    bordered_frames = get_frames(BBoxPartition, reBBoxSize, local_time_interval, config, evalscript, bordered=True)
    print(bordered_frames[0].shape, len(bordered_frames))
    # Reshape the list of frames into a 2D grid
    grid_frames = np.array(bordered_frames[::-1]).reshape((N, M, *bordered_frames[0].shape))
    print(grid_frames.shape)
    # Concatenate along the rows to merge frames within each row
    merged_columns = [np.concatenate(column, axis=0) for column in grid_frames]
    print(merged_columns[0].shape)
    # Concatenate the rows vertically to form the final merged image
    merged_image = np.concatenate(merged_columns[::-1], axis=1)
    print(merged_image.shape)
    ax[0].imshow(merged_image)
    ax[0].set_title(f"Patched image")
    
    ax[1].imshow(og_image[-1])
    ax[1].set_title(f"Original image w. refitted bbox")
    fig.suptitle("TITLE")
    fig.tight_layout(rect=[0,0,0,0])

    plt.show()
    return merged_image


def ImagePlot(BBCoords, config, resolution, date, zoomfactor: Optional[float] = 1, dpi: Optional[int] = 96, evalscript: Optional[str] = evalscripts["evalscript1"], datacollection: Optional[DataCollection] = DataCollection.SENTINEL1):
    CoordBBox = BBox(bbox=BBCoords, crs=CRS.WGS84)                     #Setup bbox, crs is a kind of coordinate reference system   
    ogBBoxSize = bbox_to_dimensions(CoordBBox, resolution = resolution)
    _, ax = plt.subplots(dpi=dpi, frameon=False)
    
    local_time_interval = date, date

    frame = collect_data(local_time_interval, CoordBBox,
                            ogBBoxSize, config, evalscript, datacollection)


    ax.imshow(zoom(frame[-1], x=np.ceil(len(frame[-1][0,:,0])/2), y=np.ceil(len(frame[-1][:,0,0])/2), zoom=zoomfactor))
    ax.axis('off')

    print(f"Single element in the list is of type {type(frame[-1])} and has shape {frame[-1].shape}")

def merge_patches(patches, N, M):
    # Reshape the list of frames into a 2D grid
    print(np.array(patches[::-1]).shape, len(patches))
    grid_frames = np.array(patches[::-1]).reshape((N, M, *patches[0].shape))
    print(grid_frames.shape)
    # Concatenate along the rows to merge frames within each row
    merged_columns = [np.concatenate(column, axis=0) for column in grid_frames]
    print(merged_columns[0].shape)
    # Concatenate the rows vertically to form the final merged image
    merged_image = np.concatenate(merged_columns[::-1], axis=1)
    return merged_image

def merge_patchesv2(image, patch_size, model):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #print(image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.figure()

    bp =BatchPatching(patchsize=patch_size[0], overlap=0.2, stride=None, typ='torch')
    x = transforms.ToTensor()(image)
    x = torch.unsqueeze(x, dim=0)
    batch_patches, batch_indices = bp.patch_batch(x)
    for i, patch in enumerate(batch_patches[0]):
        patch = torch.unsqueeze(patch.permute(2,0,1), dim=0)
        m = model(patch.to(Config.device)).detach().cpu()
        batch_patches[0][i] = torch.squeeze(m, dim=0).permute(1,2,0)       
        
    merged_batch = bp.merge_batch(batch_patches, batch_indices, mode='avg')
    merged_batch = transforms.ToTensor()(np.squeeze(merged_batch, axis=0))
    merged_batch = torch.unsqueeze(merged_batch.permute(1,0,2), dim=0)
    predicted_classes = torch.argmax(merged_batch, dim=1)
    mask = torch.zeros_like(merged_batch)

    mask.scatter_(1, predicted_classes.unsqueeze(1), 1)

    # Assign values based on the predicted class
    background_value = 0
    oil_spill_value = 1
    look_alike_value = 2
    ships_value = 3
    land_value = 4

    # Assign values based on the predicted class
    result_image = (
        mask[:, 0:1, :, :] * background_value +
        mask[:, 1:2, :, :] * oil_spill_value +
        mask[:, 2:3, :, :] * look_alike_value +
        mask[:, 3:4, :, :] * ships_value +
        mask[:, 4:5, :, :] * land_value
    )

    return result_image.cpu().numpy().squeeze()

"""merged_predictions.append(merge_patchesv2(merged_images[i], patch_size, model))"""
def merge_patchesv3(image, patch_size, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictions_smooth = predict_img_with_smooth_windowing(
        image,
        window_size=patch_size[0],
        subdivisions=int(2),  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=5,
        pred_func=model
        ).transpose(2,0,1)

    print(predictions_smooth.shape)
    predicted_classes = torch.argmax(torch.FloatTensor(predictions_smooth).unsqueeze(0), dim=1)
    mask = torch.zeros_like(torch.FloatTensor(predictions_smooth).unsqueeze(0))

    mask.scatter_(1, predicted_classes.unsqueeze(1), 1)

    # Assign values based on the predicted class
    background_value = 0
    oil_spill_value = 1
    look_alike_value = 2
    ships_value = 3
    land_value = 4

    # Assign values based on the predicted class
    result_image = (
        mask[:, 0:1, :, :] * background_value +
        mask[:, 1:2, :, :] * oil_spill_value +
        mask[:, 2:3, :, :] * look_alike_value +
        mask[:, 3:4, :, :] * ships_value +
        mask[:, 4:5, :, :] * land_value
    )

    plt.imshow(result_image.squeeze(), cmap='jet', vmin=0, vmax=4)  # Using 'jet' colormap with values ranging from 0 to 4
    plt.colorbar(ticks=[0, 1, 2, 3, 4])  # Add colorbar with ticks for each class
    plt.show()
    return result_image.cpu().numpy().squeeze()

def resize_predict(image, im_size, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (672,1280))
    image = transforms.ToTensor()(image)
    image = torch.unsqueeze(image, dim=0)
    #print(image.size())
    m = model(image.to(Config.device)).detach().cpu()
    m = transforms.Resize((im_size[0], im_size[1]))(m)
    #print(m.size())
    predicted_classes = torch.argmax(m, dim=1)
    mask = torch.zeros_like(m)
    #print(predicted_classes.size())
    mask.scatter_(1, predicted_classes.unsqueeze(1), 1)

    # Assign values based on the predicted class
    background_value = 0
    oil_spill_value = 1
    look_alike_value = 2
    ships_value = 3
    land_value = 4

    # Assign values based on the predicted class
    result_image = (
        mask[:, 0:1, :, :] * background_value +
        mask[:, 1:2, :, :] * oil_spill_value +
        mask[:, 2:3, :, :] * look_alike_value +
        mask[:, 3:4, :, :] * ships_value +
        mask[:, 4:5, :, :] * land_value
    )

    """plt.imshow(result_image.squeeze(), cmap='jet', vmin=0, vmax=4)  # Using 'jet' colormap with values ranging from 0 to 4
    plt.colorbar(ticks=[0, 1, 2, 3, 4])  # Add colorbar with ticks for each class
    plt.show()"""
    return result_image.cpu().numpy().squeeze()

def improved_patching(BBCoords, dates, resolution, config, model, im_size: Optional[Tuple[int,int]] = (650, 1250), patch_size: Optional[Tuple[int,int]] = (320,320), datacollection: Optional[DataCollection] = DataCollection.SENTINEL1, dpi: Optional[int] = 96, evalscript: Optional[str] = evalscripts["evalscript1"]):
    dates = list(set(dates))
    total_possible_days = len(dates)                                    #Total amount of days available
    
    #######
    CoordBBox = BBox(bbox=BBCoords, crs=CRS.WGS84)                     #Setup bbox, crs is a kind of coordinate reference system          
    BBoxSize = bbox_to_dimensions(CoordBBox, resolution = resolution)  #Size of the bbox    

    CoordsUtm1 = transform_point((BBCoords[0], BBCoords[1]),source_crs = CRS.WGS84,
                             target_crs = get_utm_crs(BBCoords[0], BBCoords[1]))
    CoordsUtm2 = transform_point((BBCoords[2], BBCoords[3]),source_crs = CRS.WGS84,
                                 target_crs = get_utm_crs(BBCoords[2], BBCoords[3]))

    UTMBBox = BBox(bbox=(CoordsUtm1[0], CoordsUtm1[1], CoordsUtm2[0], CoordsUtm2[1]), crs = get_utm_crs(BBCoords[0], BBCoords[1]))
    utm_BBoxSize = bbox_to_dimensions(UTMBBox, resolution = resolution)

    Distance_utmx, Distance_utmy = abs(CoordsUtm1[0] - CoordsUtm2[0]), abs(CoordsUtm1[1] - CoordsUtm2[1])

    partition_size_x = Distance_utmx * ((im_size[1]/utm_BBoxSize[0]))   
    partition_size_y = Distance_utmy * ((im_size[0]/utm_BBoxSize[1]))  #Lat / Long coordinates difference and resolution scale.

    BBoxPartition = UTMBBox.get_partition(size_x=partition_size_x,
                                            size_y=partition_size_y)  #Partition of the bbox for the data collection

    for i in range(len(BBoxPartition)):
        for j in range(len(BBoxPartition[0][:])):
            BBoxPartition[i][j] = BBox.transform(BBoxPartition[i][j], crs = CRS.WGS84)
    
    res_up = bbox_to_resolution(BBoxPartition[0][0], width=im_size[1], height=im_size[0])     #Resolution update to end up with 1250x650 image
    reBBoxSize = bbox_to_dimensions(BBoxPartition[0][0], resolution = res_up)                       #Size of the bbox at the updated resolution

    ReBBCoords = (BBoxPartition[0][-1].lower_left[0],
                    BBoxPartition[0][-1].lower_left[1],
                    BBoxPartition[-1][0].upper_right[0],
                    BBoxPartition[-1][0].upper_right[1])
    
    CoordReBBox = BBox(bbox=ReBBCoords, crs=CRS.WGS84)                     #Setup bbox, crs is a kind of coordinate reference system        
    ogReBBoxSize = bbox_to_dimensions(CoordReBBox, resolution = res_up)    #Size of the bbox at the updated resolution

    print(f"\nImage shape at {resolution} m resolution: {BBoxSize} pixels\n")
    print(f'\nResized image shape at {res_up} m resolution: {ogReBBoxSize} pixels\n')
    print(f"\nResized image shape at ")
    print(f'Partition size: {reBBoxSize} pixels, with resolution {res_up} m and shape {len(BBoxPartition)} x {len(BBoxPartition[0][:])} partitions.\n')
    #print(f'resolution error percentage: in x {round(abs(res_up[0]/resolution*100-100),2)}%, in y: {(round(abs(res_up[1]/resolution*100-100),2))} %\n')
    print(f'Original bbox coordinates: {BBCoords}, Partioned BBox coordinates: {BBoxPartition[0][0].upper_right[0],BBoxPartition[0][0].upper_right[1], BBoxPartition[-1][-1].lower_left[0],BBoxPartition[-1][-1].lower_left[1]}\n')

    images = []
    num_it =0
    for date in dates:
        date_batch = []
        for i in range(len(BBoxPartition)):
            for j in range(len(BBoxPartition[0][:])):
                date_batch.append(collect_data(date, BBoxPartition[i][j], reBBoxSize, config, evalscript=evalscript, datacollection=datacollection)[-1])
        images.append(date_batch)
    
    #print(len(images), len(images[0]))
    #print("Shown image shape: ", np.array(images[0]).shape)
    #plt.imshow(images[4][0])
    #Indeks: images[DATE][LARGE PATCH][-1]

    file_path = os.getcwd()
    merged_images = []
    merged_predictions = []
    
    for i in range(len(images)):
        num_im=0
        date_name = dates[i]
        merged_images.append(merge_patches(images[i][:], len(BBoxPartition), len(BBoxPartition[0][:])))
        temp_img = []
        for image in images[i]:
            image = cv2.resize(image, (1250,650))
            num_im += 1
            _, ax = plt.subplots(dpi=300, frameon=False)
            ax.imshow(image)
            ax.axis('off')
            plt.imsave(f"{file_path}/output/test/{date_name}_{num_im}.png", image)
            plt.figure()
"""            
            #temp_img.append(merge_patchesv2(image, patch_size, model))
            temp_img.append(resize_predict(image, im_size, model))

        merged_predictions.append(merge_patches(temp_img, len(BBoxPartition), len(BBoxPartition[0][:])))
        cmap = mcolors.ListedColormap(['black', 'blue', 'maroon', 'wheat', 'darkgreen'])
        norm = Normalize(vmin=0.0, vmax=4.0)
        _, ax = plt.subplots(dpi=300, frameon=False)
        ax.imshow(merged_predictions[i], cmap=cmap, norm=norm)
        ax.axis('off')
        plt.figure()
        ######
    return merged_images, merged_predictions"""

def create_temporal_hist(merged_predictions, dates, include_cold_periods: Optional[bool]=True, PlotDates: Optional[list]=None):
    years = np.unique([date[0:3] for date in dates])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days_with_possible_oil = []
    if len(years) == 1:
        fig, ax = plt.subplots(len(years), 1, figsize=(10, 6))
    else:
        fig, ax = plt.subplots(len(years), 1, figsize=(10, 3*len(years)))
                               
    if include_cold_periods == False:
        for date in dates:
            if str(date[5:7]) in ["01", "02", "03", "04", "05", "11", "12"]:
                dates.remove(date)
                
    # Iterate over each year
    for i, year in enumerate(years):
        # Filter data for the current year
        year_indices = [i for i, date in enumerate(dates) if date.startswith(year)]
        year_predictions = np.flip([merged_predictions[i] for i in year_indices])
        year_images = None
        high_oil_days = []
        high_oil_predictions = []
        if PlotDates is not None:
            year_images = np.flip([PlotDates[i] for i in year_indices])
        year_dates = np.flip([dates[i] for i in year_indices])

        # Initialize lists to store oil percentages and month ticks
        oil_percentages = []
        month_ticks = []
        current_month = None

        # Accumulate data for the current year
        e = 0
        for date, prediction in zip(year_dates, year_predictions):
            unique, counts = np.unique(prediction, return_counts=True)
            if 1 in unique:
                average_oil = (counts[1] / np.sum(counts))*100
            else:
                average_oil = 0
            if average_oil > 2:
                days_with_possible_oil.append(date)
                if year_images is not None:
                    high_oil_predictions.append(prediction)
                    high_oil_days.append(year_images[e])
            oil_percentages.append(average_oil)

            # Group dates by month
            month = date[5:7]
            if month != current_month:
                current_month = month
                month_ticks.append(date)
            e += 1
        if len(years) != 1:
            # Create subplot for the current year
            ax[i].bar(year_dates, oil_percentages, color='skyblue')

            ax[i].set_xticks(month_ticks)
            ax[i].set_xticklabels([date[8:10]+"-"+str(months[int(date[5:7])-1]) for date in month_ticks], rotation=45)
            ax[i].set_ylim(0, 10)
            ax[i].set_xlabel('Date')
            ax[i].set_ylabel('Percentage of Oil Spill Pixels')
            ax[i].set_title(f'Oil Spill Percentage Over Time - {year}')
        else:
            # Create subplot for the current year
            ax.bar(year_dates, oil_percentages, color='skyblue')

            ax.set_xticks(month_ticks)
            ax.set_xticklabels([date[8:10]+"-"+str(months[int(date[5:7])-1]) for date in month_ticks], rotation=45)
            ax.set_ylim(0, 10)
            ax.set_xlabel('Date')
            ax.set_ylabel('Percentage of Oil Spill Pixels')
            ax.set_title(f'Oil Spill Percentage Over Time - {year}')
    fig.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
    print(days_with_possible_oil)
    return high_oil_days, high_oil_predictions




# Plotting


#TEST OF VISUALIZE_PATCHES

"""access_token = get_access_token(
"olivermillinge@live.dk",
getpass.getpass("Enter your password: "),
)
0
setup_config(profile="OliverML",
              client="sh-f74155bc-d801-4214-b465-91dccbf00e36",
              secret="dOtgNKKlBsQYry8Gxt3u78WvgiqzvDsI")"""

config = Connect(profile="OliverML")
        
#BBCoords = (-17.237549,75.696646,-15.930176,74.99641)
BBCoords = (9.624023,78.297158,11.167603,78.499317)
#BBCoords = (9.788818,78.176211,12.205811,78.650023)
#BBCoords = (10.420532,77.930610,12.601318,78.637584)
ImWidth, ImHeight = 650, 1250                                       #Image width and height
resolution = (40,40)                                                #Resolution of the data
time_interval = "2018-05-22", "2018-05-22"                          #Time interval for the data
datacollection = DataCollection.SENTINEL1_EW                        #Data collection to be used: s1, s1ew
mydpi = 300                                                         #DPI for plotting the images        

file_path = os.getcwd()
model = torch.load(os.path.join(file_path, Config.dbg_model)).to(Config.device)
dates = extract_possible_dates(time_interval, BBCoords, config, datacollection)
"""colddates = dates.copy()
for date in dates:
    if str(date[5:7]) in ["01", "02", "03", "04", "05", "11", "12"]:
        colddates.remove(str(date))
print("colddate: ", colddates)
print(len(colddates))"""
#plt.clf()
#plt.cla()
#ImagePlot(BBCoords, config, resolution, "2023-09-28", zoomfactor=1)
#visualize_patches(BBCoords, time_interval, mydpi, config, resolution)

#print(len(dates))
#years = np.unique([date[:4] for date in dates])
#print("years: \n \n \n \n HERE \n \n \n ", years)

merged_images, merged_predictions = improved_patching(BBCoords, dates, resolution, config, model, datacollection=datacollection, dpi=mydpi, evalscript=evalscripts["evalscript2"])

"""file_path = os.path.dirname(os.path.realpath(__file__))
imagepath = "C:/Users/Mr. Oliver/Desktop/CNN/FinalModels/UNet/output/test"
images = sorted(list(paths.list_images(imagepath)))
images_arrays = []
for image in images:
    images_arrays.append(cv2.imread(image))

merged_images = []
merged_predictions = []

def merge_consecutive_images(images):
    merged_images = []
    num_images = len(images)
    
    # Iterate over pairs of consecutive images
    temp_img = []
    for i in range(int(num_images)):
        if i % 2 == 0 and i != 0:
            merged_images.append(temp_img)
            temp_img = []
        temp_img.append(images[i])
    merged_images.append(temp_img)
    return merged_images

merged_images = merge_consecutive_images(images_arrays)
print("Merged_image length: ",len(merged_images))
#print(len(merged_images))
#print(len(merged_images[0]))
for i in range(len(merged_images)):
    temp_img = []
    for image in merged_images[i]:
        print("image: ",image.shape)
        image = cv2.resize(image, (1250,650))
        #_, ax = plt.subplots(dpi=300, frameon=False)
        #ax.imshow(image)
        #ax.axis('off')
        #plt.figure()
        
        #temp_img.append(merge_patchesv2(image, patch_size, model))
        temp_img.append(resize_predict(image, (650,1250), model))
    #print(temp_img[:][0].shape)
    #print(temp_img)
    merged_predictions.append(merge_patches(temp_img, 1, 2))
    cmap = mcolors.ListedColormap(['black', 'blue', 'maroon', 'wheat', 'darkgreen'])
    norm = Normalize(vmin=0.0, vmax=4.0)
    _, ax = plt.subplots(dpi=300, frameon=False)
    ax.imshow(merged_predictions[i], cmap=cmap, norm=norm)
    ax.axis('off')
    plt.figure()


create_temporal_hist(merged_predictions, dates)"""