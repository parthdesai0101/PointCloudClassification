import numpy as np
import torch
import trimesh
import glob
import os
import sys

def load_file_data(dir_name, sample_amount = 0, normalize='none'):
    # From the ModelNet40 unzipped directory load all the data along with classes.
    # The inputs are:
    # 1. dir_name: directory which contains the ModelNet40 data
    # 2. sampleAmount: number of points to be sample from each point-cloud (if 0 then all points are taken);
    # 3. normalize: either 'none' or 'max_distance'. Max_distance will center the
    #               data about the origin (subtracting centroid from all points). 'max_distance' is a
    #               common convention for point-cloud data, making certain that the maximum distance point from
    #               the origin is 1 (all points are within a sphere of radius 1, center (0,0,0)).

    train_data = []
    train_class = []
    val_data = []
    val_class = []
    data_dir_tree = glob.glob(dir_name + "/*");
    data_dir_tree.sort()
    i = 0
    for data_dir in data_dir_tree:
        train_directories = glob.glob(os.path.join(data_dir, "train/*"));
        test_directories = glob.glob(os.path.join(data_dir, "test/*"));
        for dir in train_directories:
            if sample_amount == 0:
                temp_points = torch.tensor(trimesh.load(dir).vertices)
            else:
                temp_points = torch.tensor(trimesh.load(dir).sample(sample_amount))

            if normalize=='max_distance':
                # Center around origin and normalize all points to within a total distance
                # of 1 (-1 to 1) so all points will be within a sphere of radius 1
                temp_points = normalize_on_sphere(temp_points);

            train_data.append(temp_points);
            train_class.append(i)

        for dir in test_directories:
            if sample_amount == 0:
                temp_points = torch.tensor(trimesh.load(dir).vertices)
            else:
                temp_points = torch.tensor(trimesh.load(dir).sample(sample_amount))

            if normalize=='max_distance':
                # Center around origin and normalize all points to within a total distance
                # of 1 (-1 to 1) so all points will be within a sphere of radius 1
                temp_points = normalize_on_sphere(temp_points);


            val_data.append(temp_points);
            val_class.append(i)
        i += 1;

    train_data_tensor = torch.stack(train_data);
    val_data_tensor = torch.stack(val_data);
    train_class_tensor = torch.tensor(train_class);
    val_class_tensor = torch.tensor(val_class);

    torch.save(train_data_tensor, 'train_data');
    torch.save(val_data_tensor, 'val_data');
    torch.save(train_class_tensor, 'train_classes')
    torch.save(val_class_tensor, 'val_classes')

    return train_data_tensor, train_class_tensor, val_data_tensor, val_class_tensor

def normalize_on_sphere(data):
    mean_val = torch.mean(data, dim=0);
    data -= mean_val;
    data /= torch.max(torch.sqrt(torch.sum(data**2, axis=1)));
    return data


def normalize_on_cube(data):
    data -= torch.min(data, dim=0).values;
    data /= torch.max(data, dim=0).values;
    data = data * 2 - 1;
    data /= torch.max(torch.sqrt(torch.sum(data**2, axis=1)));
    return data

#train_data, train_class, val_data, val_class = load_file_data('ModelNet40\ModelNet40', normalize="max_distance", sample_amount=2048);


def load_saved_pointclouds():
    train_data = torch.load('train_data');
    val_data = torch.load('val_data');
    train_classes = torch.load('train_classes')
    val_classes = torch.load('val_classes')
    return train_data, val_data, train_classes, val_classes

