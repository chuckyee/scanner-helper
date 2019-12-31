#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import itertools

import numpy as np
from skimage import img_as_ubyte
from skimage import io, color, measure, transform
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


def compute_tolerance(contour, tolerance_frac):
    '''Compute tolerance for Douglas-Peucker algorithm used in
    approximate_polygon() by computing max dimension of contour in vertical and
    horizontal directions.
    '''
    coord1, coord2 = contour.T
    dim1 = np.max(coord1) - np.min(coord1)
    dim2 = np.max(coord2) - np.min(coord2)
    return tolerance_frac * max(dim1, dim2)


def points_on_line(v1, v2, points, tolerance=1):
    '''Count number of points which lies within tolerance of line defined by two
    points v1 and v2.
    '''
    v = v2 - v1
    x = points - v1
    distances = np.cross(x, v) / np.linalg.norm(v)
    return np.sum(np.abs(distances) < tolerance)


def polygon_area(points):
    '''Shoelace formula. Points must be oriented.'''
    x, y = np.asarray(points).T
    return 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))


def orient_ccw(points):
    points = np.asarray(points)
    center = np.mean(points, axis=0)
    rays = points - center
    x, y = rays.T
    theta = np.arctan2(y, x)
    # negate theta because y-axis increases _downward_
    return points[np.argsort(-theta)]


def select_rectangle(polygon, contour, min_area=0.05):
    # keep only unique points in contour and polygon
    contour = contour[:-1]      # last point is same as first; delete
    polygon = np.unique(polygon, axis=0)
    area0 = polygon_area(polygon)

    best_score = 0
    best_rect = None
    for rect in itertools.combinations(polygon, 4):
        rect = orient_ccw(rect)
        pairs = zip(np.roll(rect, 1, axis=0), rect)
        score = sum(points_on_line(v1, v2, contour) for v1, v2 in pairs)
        area = polygon_area(rect)
        # area constraint avoids collinear quadrilaterals
        if score > best_score and area / area0 > min_area:
            best_score = score
            best_rect = rect

    return best_rect


def compute_rectangle(labels, index, tolerance_frac=0.03):
    mask = labels == index
    # find_contours returns (row, col) coords, contour winds counter-clockwise
    contours = measure.find_contours(mask, 0.5, positive_orientation='high')
    contour = sorted(contours, key=len)[-1]
    # convert from (row, col) to (x, y) coords
    # convention: top-left is origin and y increases _downwards_
    contour = np.roll(contour, 1, axis=1)
    tolerance = compute_tolerance(contour, tolerance_frac)
    polygon = measure.approximate_polygon(contour, tolerance=tolerance)
    rectangle = select_rectangle(polygon, contour)
    return rectangle


def image_dimensions(corners):
    '''Given four vertices of a rectangle, computes length of long and short edges
    by averaging the length of the two short edges, and length of the two long
    edges.
    '''
    pairs = zip(np.roll(corners, 1, axis=0), corners)
    edge_lengths = [np.linalg.norm(c2 - c1) for c1, c2 in pairs]
    short1, short2, long1, long2 = np.sort(edge_lengths)
    return (short1 + short2) / 2, (long1 + long2) / 2


def extract_photo(image, rectangle):
    # find top-left corner of rect
    center = np.mean(rectangle, axis=0)
    rays = rectangle - center
    x, y = rays.T
    is_upperleft = (x < 0) & (y < 0) # recall y-axis increases downward
    upperleft_idx, = np.argwhere(is_upperleft)[0]

    # vertex order = [upper left, lower left, lower right, upper right]
    rectangle = np.roll(rectangle, -upperleft_idx, axis=0)

    short_pixels, long_pixels = image_dimensions(rectangle)

    # determine whether left edge is long or short
    upper_left, lower_left, _, _ = rectangle
    left_length = np.linalg.norm(lower_left - upper_left)
    height, width = short_pixels, long_pixels
    if abs(long_pixels - left_length) < abs(short_pixels - left_length):
        height, width = long_pixels, short_pixels

    upper_left = (0, 0)
    lower_left = (0, height)
    lower_right = (width, height)
    upper_right = (width, 0)
    src = np.array([upper_left, lower_left, lower_right, upper_right])
    dst = rectangle

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, dst)

    output_shape = (int(height), int(width))
    warped = transform.warp(image, tform3, output_shape=output_shape)
    return warped


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
    photo_indices = select_photos(label)
    print(photo_indices)

    # find contours and convert to rectangles
    rectangles = [compute_rectangle(label, index) for index in photo_indices]
    print(rectangles)

    # extract the image content within each rectangle
    photos = [extract_photo(image, rect) for rect in rectangles]

    return photos


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
                savename = append_suffix(os.path.basename(filename), n)
                path = os.path.join(args.outdir, savename)
                io.imsave(path, img_as_ubyte(photo))
            print('done.')
        except Exception as ex:
            print('FAILED.')
            print(ex)
            os.makedirs(args.errdir, exist_ok=True)
            path = os.path.join(args.errdir, os.path.basename(filename))
            io.imsave(path, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images', type=str, nargs='+',
                        help='scanned image files to process')
    parser.add_argument('--outdir', type=str, default='photos',
                        help='path to write extracted photos')
    parser.add_argument('--errdir', type=str, default='errors',
                        help='path to write photos which could not be processed')

    args = parser.parse_args()

    main(args)
