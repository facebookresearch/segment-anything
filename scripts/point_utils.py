from scipy.ndimage import gaussian_filter
import numpy as np
from utils import \
    argmax_dist, mask_out


def positive_center_point(mask, class_number):
    filter_center_candidate = gaussian_filter(mask == class_number, 10)
    #  Precise positive
    [row, col] = np.argwhere(filter_center_candidate > 0.0)[0]
    return row, col


def positive_random_point(mask, class_number, center: list = None):
    [row, col] = center
    positive = np.argwhere((mask == class_number).astype(np.int16) > 0.0)
    choices = np.random.RandomState(seed=1810).choice(np.arange(
        positive.shape[0]),
                                                      size=10)
    c = argmax_dist(positive[choices], row, col)
    positive = positive[c:c + 1][:, ::-1]
    return positive


def negative_random_with_constrain(mask):
    filter_negative = gaussian_filter(mask == 0, 3)
    filter_negative = mask_out(filter_negative,
                               xmin=200,
                               xmax=450,
                               ymin=100,
                               ymax=450,
                               to_value=0.0)
    negative = np.argwhere(filter_negative > 0.0)
    choices = np.random.RandomState(seed=1810).choice(np.arange(
        negative.shape[0]),
                                                      size=1)
    # Pickup the choice and swap row with col
    negative = negative[choices][:, ::-1]
    return negative


def make_point_from_mask(mask, class_number):
    coors = np.argwhere(mask == class_number)

    if coors.shape[0] == 0: 
        return None, None

    row, col = positive_center_point(mask, class_number)
    positive = positive_random_point(mask, class_number, [row, col])
    negative = negative_random_with_constrain(mask)

    # Make the col/row and label
    coors = np.array([[col, row], *positive, *negative])
    label = np.array(
        [1, *np.ones(positive.shape[0]), *np.zeros(negative.shape[0])])

    # In Cartesian Coordinate, number of row is y-axis
    return coors, label

