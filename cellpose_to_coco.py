import numpy as np
import cv2
import os
import json
from tqdm import trange

def main():
    # modify these:
    data_dir = 'data'
    save_dir = 'coco2'
    date_captured = '2022-01-11'
    info = {
        "year": "2022",
        "version": "1.0",
        "description": "U3D_201910_2022-01-11_23-11-36.799392 dataset",
        "contributor": "Prakas Lab",
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
    # initialize images, annotations
    images = []
    annotations = []
    # load files
    npy_files = [os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.endswith('.npy')]
    # set up coco-compliant file structure
    imgs_dir = "data"
    os.makedirs(os.path.join(save_dir, imgs_dir), exist_ok = True)

    ann_idx = 0
    # for each view, load it and save the original image
    for i, f in enumerate(npy_files):
        fname = f.split('.')[0]
        fname = fname.split('/')[-1]
        items = np.load(f, allow_pickle=True).item()
        mask = (items['masks'][:, :, None])
        image = items['img']

        cv2.imwrite(os.path.join(save_dir, imgs_dir, fname + ".png"), image)

        # save image data - all have the same license and date captured
        image_data = {
            "id": i,
            "license": 0,
            "width": image.shape[0],
            "height": image.shape[1],
            "file_name": fname + ".png",
            "date_captured": date_captured 
        }
        images.append(image_data)

        # for each item in the mask, make a bounding box and bounding polygon
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
                print(f'view {f} cell {cell} has a bad polygon - {len(poly)}, contour shape {[contour.shape for contour in contours]}')
                impath = os.path.join(f'{f}_{cell}_mask_bad.png')
                cell_mask2 = cell_mask.copy()
                cv2.drawContours(cell_mask2, contours_poly, -1, (150,150,150), 3)
                cv2.imwrite(impath, cell_mask2)
                expansion = 5
                kernel = np.zeros((expansion,expansion),np.uint8)
                kernel = cv2.circle(kernel, (int(expansion/2), int(expansion/2)), int(expansion/2), (255,255,255), -1)
                cell_mask = cv2.dilate(cell_mask,kernel,iterations = 4)
                view = image.copy()
                view[cell_mask == 0] = 0
                impath = os.path.join(f'{f}_{cell}_cell_bad.png')
                cv2.imwrite(impath, view)
                continue
           
            annotation = {
                "id": ann_idx,
                "image_id": i,
                "category_id": 0,
                "segmentation": [poly],
                "bbox": bounding_box,
                "area": area
            }

            ann_idx += 1

            annotations.append(annotation)

    # save all to a single dict
    coco_annotations = {"info": info, "licenses": licenses, "categories": categories, "images": images, "annotations": annotations}
    # save dict to file as json
    with open(os.path.join(save_dir, "labels.json"), 'w') as f:
        f.write(json.dumps(coco_annotations))

if __name__ == '__main__':
    main()