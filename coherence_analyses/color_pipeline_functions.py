import os
import numpy as np
import scipy.stats
from PIL import Image
from scipy.spatial import ConvexHull


def transform_images_from_RGB_to_jzazbz(rgb_array, jzazbz_array):
    """
    Converts rgb pixel values to JzAzBz pixel values
    Args:
        rgb_array (array): matrix of rgb pixel values
    Returns:
        jzazbz_arrays (array): matrix of JzAzBz pixel values
    """
    rgb_array = rgb_array[:,:,0:3]
    r = rgb_array[:,:,0].reshape([-1])
    g = rgb_array[:,:,1].reshape([-1])
    b = rgb_array[:,:,2].reshape([-1])
    jzazbz_vals = jzazbz_array[r,g,b]
    jzazbz_coords = jzazbz_vals.reshape(list(rgb_array.shape[:3])).transpose([0,1,2])
    return jzazbz_coords


def initialize_image_directory(data_path):
    ###
    image_directory = {}
    ###
    for directory in os.listdir(data_path):
        image_directory[directory] = []
        for file in os.listdir(data_path+'{}'.format(directory)):
            if file.endswith('.png'):
                image_directory[directory].append(file)
        image_directory[directory] = np.ravel(image_directory[directory])
    return image_directory


def get_image_data(image_directory, data_path, jzazbz_array):
    image_data = {}
    ###
    jzazbz_stripe_arrays = {}
    jz_stripe_means = {}
    az_stripe_means = {}
    bz_stripe_means = {}
    r_stripe_means = {}
    g_stripe_means = {}
    b_stripe_means = {}
    ###
    for key in image_directory.keys():
        for img in image_directory[key]:
            img_temp = np.array(Image.open(data_path+'{}/{}'.format(key,img)))
            ###
            r_stripe_means[img] = np.mean(img_temp[:,:,0])
            g_stripe_means[img] = np.mean(img_temp[:,:,1])
            b_stripe_means[img] = np.mean(img_temp[:,:,2])
            ###
            img_jzazbz = transform_images_from_RGB_to_jzazbz(img_temp, jzazbz_array)
            jz_stripe_means[img] = np.mean(img_jzazbz[:,:,0])
            az_stripe_means[img] = np.mean(img_jzazbz[:,:,1])
            bz_stripe_means[img] = np.mean(img_jzazbz[:,:,2])
            ###
            dist = np.ravel(np.histogramdd(np.reshape(img_jzazbz[:,:,:],((img_jzazbz.shape[0])*(img_jzazbz.shape[1]),3)), 
                                          bins=(np.linspace(0,0.167,3),np.linspace(-0.1,0.11,3),
                                               np.linspace(-0.156,0.115,3)), density=True)[0])
            jzazbz_stripe_arrays[img] = dist
            ###
    image_data['jzazbz_stripe_arrays'] = jzazbz_stripe_arrays
    image_data['jz_stripe_means'] = jz_stripe_means
    image_data['az_stripe_means'] = az_stripe_means
    image_data['bz_stripe_means'] = bz_stripe_means
    image_data['r_stripe_means'] = r_stripe_means
    image_data['g_stripe_means'] = g_stripe_means
    image_data['b_stripe_means'] = b_stripe_means
    ###
    return image_data


def get_rgb_list(image_data):
    rgb_list = []
    for key in image_data['az_stripe_means'].keys():
        rgb_list.append(np.array([image_data['r_stripe_means'][key],
                                  image_data['g_stripe_means'][key],
                                  image_data['b_stripe_means'][key]]))
    return rgb_list


def measure_color_coherence(image_data, image_directory):
    ###
    jzazbz_coherence = {}
    jzazbz_coherence_all = []
    fail = 0
    ###
    for key in image_directory.keys():
        jzazbz_coherence[key] = []
        for img1 in image_directory[key]:
            try:
                dist1 = image_data['jzazbz_stripe_arrays'][img1]
            except:
                fail += 1
            for img2 in image_directory[key]:
                if img2 != img1:
                    try:
                        dist2 = image_data['jzazbz_stripe_arrays'][img2]
                        mean = (dist1+dist2)/2.
                        jzazbz_coherence[key].append((scipy.stats.entropy(dist1,mean)+scipy.stats.entropy(dist2,mean))/2.)
                        jzazbz_coherence_all.append((scipy.stats.entropy(dist1,mean)+scipy.stats.entropy(dist2,mean))/2.)
                    except:
                        fail += 1
    jzazbz_coherence_all = np.ravel(jzazbz_coherence_all)[np.ravel(jzazbz_coherence_all)>1e-300]
    return jzazbz_coherence, jzazbz_coherence_all


def get_convex_hulls(image_data, image_directory):
    hulls = []
    for key in image_directory.keys():
        try:
            vals = []
            for img in image_directory[key]:
                vals.append(np.array([image_data['az_stripe_means'][img],
                                      image_data['bz_stripe_means'][img],
                                      image_data['jz_stripe_means'][img]]))
            hull = ConvexHull(vals)
            hulls.append(hull)
        except:
            continue
    return hulls


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def color_coherence_fraction(image_data, hulls):
    num = 0
    for p in np.array([list(image_data['az_stripe_means'].values()),
                       list(image_data['bz_stripe_means'].values()),
                       list(image_data['jz_stripe_means'].values())]).T:
        p_bools = []
        for hull in hulls:
            p_bools.append(point_in_hull(p,hull))
        p_bools = np.array(p_bools)
        if len(p_bools[p_bools==True]) == 1:
            num+=1
    return num/(1.*len(np.array([list(image_data['az_stripe_means'].values()),
                                 list(image_data['bz_stripe_means'].values()),
                                 list(image_data['jz_stripe_means'].values())]).T))