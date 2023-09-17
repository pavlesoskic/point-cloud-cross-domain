import json
import os
import numpy
import math
import glob

SPACE = [0, -40, -3, 70.4, 40, 1]

labels_path = 'OpenPCDet/data/custom/labels'
points_path = 'OpenPCDet/data/custom/points'
imagesets_path = 'OpenPCDet/data/custom/ImageSets'

LABEL_LIMITS = []
for path in [labels_path, points_path, imagesets_path]:
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)

once_test_info_path = 'data/once/test_infos/data'
once_train_info_path = 'data/once/train_infos/data'
once_val_info_path = 'data/once/val_infos/data'

once_test_lidar_path = 'data/once/test_lidar/data'
once_train_lidar_path = 'data/once/train_lidar/data'
once_val_lidar_path = 'data/once/val_lidar/data'

def read_json(file_path):
    with open(file_path, 'r') as f:
        d = json.load(f)
        return d

def create_frame(info, path):
    annos = info['annos']
    label_lines = create_label_file(annos)
    if len(label_lines) == 0:
        return False
    label_lines = "\n".join(label_lines)
    with open(labels_path + '/' + info['sequence_id']+'_'+ info['frame_id']+'.txt', 'w') as f:
        f.write(label_lines)
    points = create_point_file(info, path)
    numpy.save(points_path + '/' + info['sequence_id']+'_'+ info['frame_id']+'.npy', points)
    return True

def is_bounding_box_within_space(bbox, space_limits):
    x, y, z, dx, dy, dz, theta = bbox
    min_x, min_y, min_z, max_x, max_y, max_z = space_limits
    rotated_dx = abs(dx * math.cos(theta)) + abs(dy * math.sin(theta))
    rotated_dy = abs(dx * math.sin(theta)) + abs(dy * math.cos(theta))
    rotated_dz = dz
    corners = [
        (x + rotated_dx/2, y + rotated_dy/2, z + rotated_dz/2),
        (x + rotated_dx/2, y - rotated_dy/2, z + rotated_dz/2),
        (x - rotated_dx/2, y + rotated_dy/2, z + rotated_dz/2),
        (x - rotated_dx/2, y - rotated_dy/2, z + rotated_dz/2),
        (x + rotated_dx/2, y + rotated_dy/2, z - rotated_dz/2),
        (x + rotated_dx/2, y - rotated_dy/2, z - rotated_dz/2),
        (x - rotated_dx/2, y + rotated_dy/2, z - rotated_dz/2),
        (x - rotated_dx/2, y - rotated_dy/2, z - rotated_dz/2)
    ]
    for corner in corners:
        if (min_x <= corner[0] <= max_x and
            min_y <= corner[1] <= max_y and
            min_z <= corner[2] <= max_z):
            continue
        else:
            return False

    return True

def create_label_file(anno):
    rows = []
    for label, placement in zip(anno['names'], anno['boxes_3d']):
        def transform_placement(placement):
            new_placement = []
            new_placement.append(-placement[1])
            new_placement.append(placement[0])
            new_placement.append(placement[2])
            new_placement.append(placement[3])
            new_placement.append(placement[4])
            new_placement.append(placement[5])
            new_placement.append(placement[6] + numpy.pi/2)
            return new_placement
        
        placement = transform_placement(placement)
        inside = is_bounding_box_within_space(placement, SPACE)
        if label in ['Car', 'Pedestrian', 'Cyclist'] and inside:
            row = create_label_row(placement, label)
            rows.append(row)
    return rows

def create_label_row(placement, label):
    placement_str = ' '.join([str(x) for x in placement])
    placement_str = placement_str + ' ' + label
    return placement_str

def create_point_file(info, lidar_path):
    file_path = lidar_path + '/' + info['sequence_id'] + '/lidar_roof/' + info['frame_id'] + '.bin'
    pcd = numpy.fromfile(file_path, dtype=numpy.float32).reshape(-1, 4)

    kitti_pcd = numpy.zeros_like(pcd)

    kitti_pcd[:, 0] = - pcd[:, 1]
    kitti_pcd[:, 1] =  pcd[:, 0]
    kitti_pcd[:, 2] = pcd[:, 2]
    kitti_pcd[:, 3] = pcd[:, 3]

    return kitti_pcd


if __name__ == '__main__':
    train_infos = os.listdir(once_train_info_path)
    val_infos = os.listdir(once_val_info_path)

    train_info_jsons = []
    for dir in train_infos:
        js = read_json(once_train_info_path + '/'  + dir + '/' + dir + '.json')
        train_info_jsons += js['frames']

    counter = 0
    train_split = []
    for train_info in train_info_jsons:
        if 'annos' in train_info:
            if create_frame(train_info, once_train_lidar_path):
                train_split.append(train_info['sequence_id']+'_'+ train_info['frame_id'])
                counter +=1
    train_split = '\n'.join(train_split)
    with open(imagesets_path + '/train.txt', 'w') as f:
        f.write(train_split)

    print('found train annos ', {counter})

    val_info_jsons = []
    for dir in val_infos:
        js = read_json(once_val_info_path + '/'  + dir + '/' + dir + '.json')
        val_info_jsons += js['frames']

    counter = 0
    val_split = []
    for val_info in val_info_jsons:
        if 'annos' in val_info:
            if create_frame(val_info, once_val_lidar_path):
                val_split.append(val_info['sequence_id']+'_'+ val_info['frame_id'])
                counter +=1
    val_split = '\n'.join(val_split)
    with open(imagesets_path + '/val.txt', 'w') as f:
        f.write(val_split)