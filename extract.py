#!/usr/bin/env python3
# coding: utf-8

import argparse
import os

import numpy as np
from skimage import io, color
from skimage import measure
from skimage import transform
from skimage.filters import sobel
from skimage.morphology import watershed
import scipy.ndimage as ndi



def select_photos(labels, min_area=0.05):
    values, counts = np.unique(labels, return_counts=True)

    ypixels, xpixels = labels.shape
    total_pixels = ypixels * xpixels

    # remove dust and margins: filter using area thresholding
    kept = [(v, area) for v, area in zip(values, counts)
            if area / total_pixels >= min_area]
    values, counts = np.array(kept).T

    # remove background: reject areas touching image edge
    edges = np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1]))
    edge_labels = np.unique(edges)

    photo_labels = [v for v in values if v not in edge_labels]

    return np.array(photo_labels)


def filter_ends(corners):
    if len(corners) == 5:
        return corners[:-1]
    elif len(corners) == 6:
        return corners[1:-1]
    else:
        raise Exception("Cannot filter ends from polygon")


def image_dimensions(corners):
    extended = np.vstack((corners, [corners[0]]))
    edge_lengths = [np.linalg.norm(c2 - c1) for c1, c2 in zip(extended[:-1], extended[1:])]
    short1, short2, long1, long2 = np.sort(edge_lengths)
    return (short1 + short2) / 2, (long1 + long2) / 2


def extract_photos(image, polygons):
    photos = []
    for polygon in polygons:
        # convert from (row, col) to (x, y) coords
        # convention: top-left is origin and y increases _downwards_
        polygon = np.roll(polygon, 1, axis=1)

        # find top-left corner of rect
        center = np.mean(polygon, axis=0)
        rays = polygon - center
        is_topleft = (rays[:,0] < 0) & (rays[:,1] < 0)
        topleft_idx, = np.argwhere(is_topleft)[0]

        # roll indices so order is [top-left, top-right, bottom-right, bottom-left]
        polygon = np.roll(polygon, -topleft_idx, axis=0)

        short, long = image_dimensions(polygon)

        src = np.array([(0, 0), (long, 0), (long, short), (0, short)])
        dst = polygon

        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = transform.warp(image, tform3, output_shape=(int(short), int(long)))

        photos.append(warped)

    return photos


def process_image(image):
    gray = color.rgb2gray(image)

    # Watershed transform for region detection using Sobel elevation map
    gradient = sobel(gray)

    markers = np.zeros_like(gradient)
    markers[gray > .99] = 1  # background
    markers[gray < .80] = 2  # photos

    segmentation = watershed(gradient, markers)

    label, num_features = ndi.label(segmentation - 1)

    # select photos from set of labeled regions
    photo_labels = select_photos(label)

    # find contours and convert to polygons
    polygons = []
    for n in photo_labels:
        mask = label == n
        contours = measure.find_contours(mask, 0.5)  # find_contours returns (row, col) elements
        contour = sorted(contours, key=len)[-1]
        corners = measure.approximate_polygon(contour, tolerance=50)
        corners = filter_ends(corners)
        polygons.append(corners)

    return extract_photos(image, polygons)


def append_suffix(filename, suffix):
    name, ext = os.path.splitext(filename)
    return f"{name}-{suffix}{ext}"


def main(args):
    for filename in args.images:
        try:
            os.makedirs(args.outdir, exist_ok=True)
            print(f'Processing {filename}...', end='')
            image = io.imread(filename)
            photos = process_image(image)
            for n, photo in enumerate(photos):
                savename = append_suffix(filename, n)
                io.imsave(os.path.join(args.outdir, savename), photo)
            print('done.')
        except:
            print('FAILED.')
            os.makedirs(args.errdir, exist_ok=True)
            io.imsave(os.path.join(args.errdir, filename), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images', type=str, nargs='+',
                        help='scanned image files to process')
    parser.add_argument('--outdir', type=str, default='./photos',
                        help='path to write extracted photos')
    parser.add_argument('--errdir', type=str, default='./errors',
                        help='path to write photos which could not be processed')

    args = parser.parse_args()

    main(args)
