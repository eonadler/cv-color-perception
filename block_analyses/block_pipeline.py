import os
import re
import json
import pickle5
import numpy as np
import pandas as pd


def get_block_data(jzazbz_array, path='block_stimuli/'):
    block_data = {'img_names': [], 'colors': [], 'jzazbz_colors': [], 'rgb_colors': [], 'jz_all': [],
                 'az_all': [], 'bz_all': []}
    ###
    for file in os.listdir(path):
        block_data['img_names'].append(file)
        color = np.array([float(re.search('\[(.*?),', file).group(1)),
                          float(re.search(',(.*?),', file).group(1)),
                          float(re.search('0(.*?)\]', file.split(' ')[-1]).group(1))])
        block_data['colors'].append(color*255.)
        block_data['rgb_colors'].append([int(round(255.*color[0])), 
                                         int(round(255.*color[1])), 
                                         int(round(255.*color[2]))])
        block_data['jzazbz_colors'].append(jzazbz_array[int(round(255.*color[0])), 
                                                        int(round(255.*color[1])), 
                                                        int(round(255.*color[2]))])
        block_data['jz_all'].append(jzazbz_array[int(round(255.*color[0])), 
                                                 int(round(255.*color[1])), 
                                                 int(round(255.*color[2]))][0])
        block_data['az_all'].append(jzazbz_array[int(round(255.*color[0])), 
                                                 int(round(255.*color[1])), 
                                                 int(round(255.*color[2]))][1])
        block_data['bz_all'].append(jzazbz_array[int(round(255.*color[0])), 
                                                 int(round(255.*color[1])), 
                                                 int(round(255.*color[2]))][2])
    ###
    length = len(block_data['jzazbz_colors'])
    block_data_stats = {'minmax_dists': np.zeros((length,length)), 'dists': np.zeros((length,length)), 
                        'minmax_rgb_dists': np.zeros((length,length)), 'rgb_dists': np.zeros((length,length)),
                       'files1': np.empty((length,length), dtype="S100"), 'files2': np.empty((length,length), dtype="S100"),
                       'colors1': np.empty((length,length,3)), 'colors2': np.empty((length,length,3)),
                       'jz_all_mean': np.zeros((length,length)), 'jz_all_diff': np.zeros((length,length)),
                       'az_all_mean': np.zeros((length,length)), 'az_all_diff': np.zeros((length,length)),
                       'bz_all_mean': np.zeros((length,length)), 'bz_all_diff': np.zeros((length,length))}
    ###
    for i,color1 in enumerate(block_data['jzazbz_colors']):
        for j,color2 in enumerate(block_data['jzazbz_colors']):
            block_data_stats['dists'][i][j] = np.sqrt((color1[0]-color2[0])**2+(color1[1]-color2[1])**2+(color1[2]-color2[2])**2)
            block_data_stats['rgb_dists'][i][j] = np.sqrt((block_data['rgb_colors'][i][0]-block_data['rgb_colors'][j][0])**2+(block_data['rgb_colors'][i][1]-block_data['rgb_colors'][j][1])**2+(block_data['rgb_colors'][i][2]-block_data['rgb_colors'][j][2])**2)
            block_data_stats['files1'][i][j] = block_data['img_names'][i]
            block_data_stats['files2'][i][j] = block_data['img_names'][j]
            block_data_stats['colors1'][i][j] = np.array(block_data['rgb_colors'][i])
            block_data_stats['colors2'][i][j] = np.array(block_data['rgb_colors'][j])
            block_data_stats['jz_all_mean'][i][j] = 0.5*(block_data['jz_all'][i]+block_data['jz_all'][j])
            block_data_stats['jz_all_diff'][i][j] = np.abs(block_data['jz_all'][i]-block_data['jz_all'][j])
            block_data_stats['az_all_mean'][i][j] = 0.5*(block_data['az_all'][i]+block_data['az_all'][j])
            block_data_stats['az_all_diff'][i][j] = np.abs(block_data['az_all'][i]-block_data['az_all'][j])
            block_data_stats['bz_all_mean'][i][j] = 0.5*(block_data['bz_all'][i]+block_data['bz_all'][j])
            block_data_stats['bz_all_diff'][i][j] = np.abs(block_data['bz_all'][i]-block_data['bz_all'][j])
    ###
    dists = np.ravel(block_data_stats['dists'])
    block_data_stats['minmax_dists'] = (dists-np.min(dists))/(np.max(dists)-np.min(dists))
    ###
    rgb_dists = np.ravel(block_data_stats['rgb_dists'])
    block_data_stats['rgb_minmax_dists'] = (rgb_dists-np.min(rgb_dists))/(np.max(rgb_dists)-np.min(rgb_dists))
    ###
    return block_data, block_data_stats


def get_block_embeddings(block_data, algorithm, path_to_embeddings, path_to_data='data_files_list.txt'):
    ###
    df = pd.read_csv(path_to_data, header=None, sep='\n')
    ###
    if algorithm=='CNN':
        with open(path_to_embeddings, 'r') as fp:
            embeddings = json.load(fp)
    elif algorithm in ['Style', 'Transformer']:
        with open(path_to_embeddings, 'rb') as fp:
            embeddings = pickle5.load(fp, encoding='unicode_escape')
    else:
        embeddings = np.loadtxt(path_to_embeddings)
    ###
    img_names_in_order = []
    for i in range(len(df.values)):
        img_names_in_order.append(df.values[i][0])
    ###
    embedding_values = []
    for img in block_data['img_names']:
        if algorithm in ['JzAzBz_wavelet', 'Grayscaled_wavelet']:
            for i in range(len(img_names_in_order)):
                if img == img_names_in_order[i]:
                    embedding_values.append(embeddings[i])
        elif algorithm == 'CNN':
            embedding_values.append(embeddings[img[0:-4]])
        elif algorithm in ['Style', 'Transformer']:
            embedding_values.append(embeddings['block_stimuli/'+img[0:-4]+'.png'])
    ###
    embedding_dists = np.zeros((len(embedding_values),len(embedding_values)))
    ###
    for i,embedding1 in enumerate(embedding_values):
        for j,embedding2 in enumerate(embedding_values):
            embedding_dists[i][j] = np.dot(embedding1,embedding2)/(np.sqrt(np.dot(embedding1,
                                                                                  embedding1)*np.dot(embedding2,
                                                                                                     embedding2)))
    ###
    embedding_dists = np.ravel(embedding_dists)
    minmax_embedding_dists = (embedding_dists-np.min(embedding_dists))/(np.max(embedding_dists)-np.min(embedding_dists))
    ###
    return minmax_embedding_dists, img_names_in_order