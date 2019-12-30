#!/usr/bin/env python3
# coding: utf-8

import argparse
import os

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


def filter_ends(corners):
    '''Filters out extraneous points from the vertex list for rectangle.

    The dense list of points generated by find_contours() has as its first and
    last entry the same "anchor" point. If this anchor point is a corner, the
    `contours` list will have 5 points:

       corners = [corner1, corner2, corner3, corner4, corner1]

    If this anchor point lies somewhere along an edge, the `contours` list will
    have 6 points:

      corners = [edge_pt, corner1, corner2, corner3, corner4, edge_pt]

    This function deletes the extraneous points.
    '''
    if len(corners) == 5:
        return corners[:-1]
    elif len(corners) == 6:
        return corners[1:-1]
    else:
        msg = f"Could not filter ends from polygon:\n{corners}"
        raise Exception(msg)


def compute_tolerance(contour, tolerance_frac):
    '''Compute tolerance for Douglas-Peucker algorithm used in
    approximate_polygon() by computing max dimension of contour in vertical and
    horizontal directions.
    '''
    coord1, coord2 = contour.T
    dim1 = np.max(coord1) - np.min(coord1)
    dim2 = np.max(coord2) - np.min(coord2)
    return tolerance_frac * max(dim1, dim2)


def compute_rectangles(labels, label_indices, tolerance_frac=0.03):
    rectangles = []
    for n in label_indices:
        mask = labels == n
        # find_contours returns (row, col) coords, contour will wind clockwise
        contours = measure.find_contours(mask, 0.5)
        contour = sorted(contours, key=len)[-1]
        tolerance = compute_tolerance(contour, tolerance_frac)
        corners = measure.approximate_polygon(contour, tolerance=tolerance)
        corners = filter_ends(corners)
        rectangles.append(corners)
    return rectangles


def image_dimensions(corners):
    '''Given four vertices of a rectangle, computes length of long and short edges
    by averaging the length of the two short edges, and length of the two long
    edges.
    '''
    extended = np.vstack((corners, [corners[0]]))
    edge_lengths = [np.linalg.norm(c2 - c1) for c1, c2 in zip(extended[:-1], extended[1:])]
    short1, short2, long1, long2 = np.sort(edge_lengths)
    return (short1 + short2) / 2, (long1 + long2) / 2


def extract_photos(image, rectangles):
    photos = []
    for rectangle in rectangles:
        # convert from (row, col) to (x, y) coords
        # convention: top-left is origin and y increases _downwards_
        rectangle = np.roll(rectangle, 1, axis=1)

        # find top-left corner of rect
        center = np.mean(rectangle, axis=0)
        rays = rectangle - center
        is_topleft = (rays[:,0] < 0) & (rays[:,1] < 0)
        topleft_idx, = np.argwhere(is_topleft)[0]

        # roll indices so order is [top-left, top-right, bottom-right, bottom-left]
        rectangle = np.roll(rectangle, -topleft_idx, axis=0)

        short_pixels, long_pixels = image_dimensions(rectangle)

        upper_left = (0, 0)
        upper_right = (long_pixels, 0)
        lower_right = (long_pixels, short_pixels)
        lower_left = (0, short_pixels)
        src = np.array([upper_left, upper_right, lower_right, lower_left])
        dst = rectangle

        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)

        output_shape = (int(short_pixels), int(long_pixels))
        warped = transform.warp(image, tform3, output_shape=output_shape)

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

    # find contours and convert to rectangles
    rectangles = compute_rectangles(label, photo_labels)

    return extract_photos(image, rectangles)


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
