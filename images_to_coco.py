import numpy as np
import cv2
import os
import json
from tqdm import trange, tqdm
from tqdm.contrib.itertools import product
import argparse
import glob
import zarr
import gcsfs
import os
import json
import time
from utils import *

from cellpose import models, io
import random

def main():
    # modify these:
    debug_mode = True
    debug_amount = 13
    cellpose_model_path = '/home/prakashlab/Documents/Kevin/Kevin/cp_dpc_new'
    gcs_project = 'soe-octopi'
    gcs_token = '/home/prakashlab/Documents/keys/data-20220317-keys.json'
    bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
    bucket_destination = 'gs://octopi-malaria-data-annotation'
    dir_out = 'coco_format_full'
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')
    flatfield_flr = np.load('flatfield_fluorescence.npy')

    # Randomly split images into training, testing, and verification datasets at this proportion
    dist = {"training": 0.7, "testing": 0.15, "verification": 0.15}
    random.seed(1)

    save_dir = 'coco_data'
    date_captured = '2022-01-11'
    info = {
        "year": "2022",
        "version": "1.0",
        "contributor": "Prakash Lab",
        "url": "https://prakashlab.stanford.edu/",
        "date_created": "2022-12-22"
    }

    licenses = [
        {
            "id": 0,
            "url": "example.com",
            "name": "placeholder"
        }
    ]

    categories = [
        {
            "id": 0,
            "name": "healthy RBC",
            "supercategory": "red blood cell"
        },
        { # not used - currently assuming all RBCs are healthy
            "id": 1,
            "name": "infected RBC",
            "supercategory": "red blood cell"
        }
    ]
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()
    # get dataset ID
    if args.data_id != None:
        DATASET_ID = [args.data_id]
    else:
        f = open('list of datasets.txt','r')
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
        f.close()
    # set info
    info["description"] = str(DATASET_ID)
    # initialize google cloud
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token
    fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
    # initialize images, annotations, dirs
    images = {}
    annotations = {}
    imgs_dir = "data"
    for key in dist.keys():
        images[key] = []
        annotations[key] = []
        os.makedirs(os.path.join(save_dir, f'{key}_{imgs_dir}'), exist_ok = True)
    # initialize cellpose
    model = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)
    # load each dataset one at a time
    parameters = {}
    for dataset_id in DATASET_ID:
        # Get acquisition parameters
        try:
            print(os.path.join(bucket_source, dataset_id, 'acquisition parameters.json'))
            json_file = fs.cat(os.path.join(bucket_source, dataset_id, 'acquisition parameters.json'))
        except:
            print("No acquisition params!")
            continue
        acquisition_parameters = json.loads(json_file)
        parameters['row_start'] = 0
        parameters['row_end'] = acquisition_parameters['Ny']
        parameters['column_start'] = 0
        parameters['column_end'] = acquisition_parameters['Nx']
        parameters['z_start'] = 0
        parameters['z_end'] = acquisition_parameters['Nz']
        if debug_mode:
            parameters['row_end'] = debug_amount
            parameters['column_end'] = debug_amount
        # initialize segmentation stats
        segmantation_stat_pd = pd.DataFrame(columns=['FOV_row','FOV_col','FOV_z','count'])
        total_number_of_cells = 0

        # iterate across each i,j,k
        ijk = product(range(parameters['row_start'],parameters['row_end']), range(parameters['column_start'],parameters['column_end']), range(parameters['z_start'],parameters['z_end']))
        for i, j, k in tqdm(ijk):
            # randomly choose which split this goes into
            r = random.random()
            a = 0
            sel_key = ""
            for key in dist.keys():
                a += dist[key]
                if r < a:
                    sel_key = key
                    break
            # get file ID
            file_id = str(i) + '_' + str(j) + '_' + str(k)
            # generate DPC
            I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
            I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
            I_FLR = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
            if len(I_BF_left.shape)==3: # convert to mono if color
                I_BF_left = I_BF_left[:,:,1]
                I_BF_right = I_BF_right[:,:,1]
            I_BF_left = I_BF_left.astype('float')/255
            I_BF_right = I_BF_right.astype('float')/255
            I_FLR = I_FLR.astype('float')/255
            # flatfield correction
            I_BF_left = I_BF_left/flatfield_left
            I_BF_right = I_BF_right/flatfield_right
            I_FLR = I_FLR/np.dstack((flatfield_flr, flatfield_flr, flatfield_flr))
            # generate dpc
            I_DPC = generate_dpc(I_BF_left,I_BF_right)
            dpc_image = I_DPC * 255
            dpc_image = dpc_image.astype("uint8")
            flr_image = I_FLR * 255
            flr_image = flr_image.astype("uint8")
            flr_image = flr_image[:,:,::-1]
            # overlay on another layer
            whole_image = np.dstack((flr_image, dpc_image))
            # store image data
            im_id = len(images[sel_key])
            image_path = os.path.join(bucket_destination, dir_out, sel_key, 'data', f'{im_id}.tiff')
            image_data = {
                "id": im_id,
                "license": 0,
                "width": I_DPC.shape[0],
                "height": I_DPC.shape[1],
                "file_name": f"{im_id}.tiff",
                "dataset": dataset_id,
                "view": file_id,
                "url": f"https://storage.cloud.google.com/{image_path[5:]}"
            }
            images[sel_key].append(image_data)
            # store image to bucket (before preprocessing)
            with fs.open(image_path, 'wb' ) as f:
                f.write(cv2.imencode('.tiff',whole_image)[1].tobytes())
            # segmentation
            # preprocessing - normalize the image
            im = I_DPC - np.min(I_DPC)
            im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
            # run segmentation
            mask, flows, styles = model.eval(im, diameter=None)
            # store stats
            number_of_cells = np.amax(mask)
            FOV_entry = pd.DataFrame.from_dict({'FOV_row':[i],'FOV_col':[j],'FOV_z':[k],'count':[number_of_cells]})
            segmantation_stat_pd = pd.concat([segmantation_stat_pd,FOV_entry])
            # cell mask polygons and bbox
            for cell in trange(1, 1 + int(np.max(mask))):
                # Filter the mask so it's only showing the cell
                cell_mask = mask.copy()
                cell_mask[cell_mask != cell] = 0
                cell_mask[cell_mask == cell] = 255
                area = np.sum(cell_mask/255)
                cell_mask = cell_mask.astype("uint8")
                contours, __ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # find polygon with the most points
                poly_list = [np.array(contour).ravel().tolist() for contour in contours]
                poly_lens = [len(i) for i in poly_list]
                i_max = poly_lens.index(np.max(poly_lens))
                contours_poly = contours[i_max]
                poly = poly_list[i_max]
                # make bounding box
                bounding_box  = cv2.boundingRect(contours_poly)
                # error handling - don't annotate if the polygon has one point
                if len(poly) <= 2:
                    continue
            
                annotation = {
                    "id": len(annotations[sel_key]),
                    "image_id": im_id,
                    "category_id": 0,
                    "segmentation": [poly],
                    "bbox": bounding_box,
                    "area": area
                }
                annotations[sel_key].append(annotation)
        # save segmentation stats for a dataset
        csv_path = os.path.join(bucket_destination, dir_out, f'{dataset_id}_segmentation_stat.csv')
        with fs.open(csv_path, 'wb') as f:
            segmantation_stat_pd.to_csv(f,index=False)
    
    # save all to a dict
    for sel_key in dist.keys():
        coco_annotations = {"info": info, "licenses": licenses, "categories": categories, "images": images[sel_key], "annotations": annotations[sel_key]}
        # save dict to file as json
        json_path = os.path.join(bucket_destination, dir_out, sel_key, 'labels.json')
        with fs.open(json_path, 'w') as f:
             f.write(json.dumps(coco_annotations))

if __name__ == '__main__':
    main()