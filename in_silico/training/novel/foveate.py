import scipy, random, os, h5py, pickle, numpy as np
from PIL import Image

def crop_image(i, params): 

    """
    'foveate' object in each image by creating a smaller bounding box around object:
    - convert metadata into pixel coordinates
    - deal with inconsistencies
    - adjust bounding box if it exceeds image dimensions
    + diameter = 50 
    """
    def check_bounding_box(min_, max_):
        """ adjust if bounding box exceeds image size"""
        if min_ < 0:
            min_, max_ = 0, window_size * 2
        if max_ > image_xy:
            min_, max_ = image_xy - (window_size * 2), image_xy
        return min_, max_

    # to align the y location between metadata adn pixels we need different scaling factors
    scale_y = {'Faces':1,'Planes':.7,'Chairs':.8,'Animals':.7,'Boats':.4,'Cars':.4,'Fruits':.7,'Tables':.7}
    scale_w = {'Faces':.8,'Planes':1,'Chairs':1.2,'Animals':1,'Boats':.9,'Cars':.8,'Fruits':1,'Tables':1}

    # get image and its metadata
    IMAGE = params['images'][i]
    image_xy = IMAGE.shape[0]
    object_size = params['sizes'][i]
    category = params['category'][i]
    object_ = params['object'][i]
    X = params['x_location'][i]
    Y = params['y_location'][i]
    # through trial and error :) 
    diameter = 50 

    # adjust windown to reflect object size
    window_size = int(diameter * (object_size + .3) * scale_w[category] )
    # align metadata with x axis image pixels
    x_center = int(-(X*int(image_xy/2)) + int(image_xy/2))
    # align metadata with y axis image pixels--scale by category specific factor
    y_center = int(-(Y*int(image_xy/2)*scale_y[category]*1) + int(image_xy/2))
    # set the y min and max of the fovea/bounding box
    min_y, max_y = y_center-window_size, y_center+window_size
    # adjust if bounding box exceeds image size
    min_y, max_y = check_bounding_box(min_y, max_y)
    # set the x min and max of the fovea/bounding box
    min_x, max_x = x_center-window_size, x_center+window_size
    # adjust if bounding box exceeds image size
    min_x, max_x = check_bounding_box(min_x, max_x)
    # generate foveated image
    foveated_image = IMAGE[min_y:max_y, min_x:max_x]

    return foveated_image


def all_stimuli(data): 
    
    params = {'images': np.array(data['images']), 
              'sizes': np.array(data['image_meta']['size']), 
              'x_location': np.array(data['image_meta']['translation_y']), 
              'y_location': np.array(data['image_meta']['translation_z']), 
              'category': np.array([i.decode() for i in data['image_meta']['category']]), 
              'object': np.array([i.decode() for i in data['image_meta']['object_name']])}

    # extract 'foveated' images
    foveated_objects = np.array([crop_image(i, params) for i in range(len(params['images']))])
    
    ## reshape to 224 x 224 
    ##foveated_objects = 
    #[np.array(Image.fromarray(i_image).resize((224, 224))) for i_image in foveated_objects] 
    
    return foveated_objects
