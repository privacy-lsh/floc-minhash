import sys
from datetime import datetime
import json
import math
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MinHash_Hierarchy.minhash import generate_target_set, generate_cryptographic_hash_list, \
    compute_hashfunc_minhash_signature, sample_random_coefficients, hash_list_from_coeffs
from pip._vendor.colorama import Fore, init

init(autoreset=True)  # to avoid reseting color everytime
from scipy import stats
from tqdm import tqdm, trange
from pathlib import Path
import matplotlib
from collections import Counter
import operator


# Dataset from kaggle https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i/overview
# downloaded from https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015#

def visualize_endpoints(train_filepath='./data/Porto/train.csv'):
    # taken from [1] https://www.kaggle.com/code/hochthom/visualization-of-taxi-trip-end-points
    # If use all the trajectories would get a heat map [2]
    # [2] https://www.kaggle.com/code/mcwitt/heatmap

    # reading training data
    # zf = zipfile.ZipFile('../input/train.csv.zip')
    # Only care about the endpoints here
    # df = pd.read_csv(zf.open('train.csv'), converters={'POLYLINE': lambda x: json.loads(x)[-1:]})

    # Polyline is a JSON array
    df = pd.read_csv(train_filepath, converters={'POLYLINE': lambda x: json.loads(x)[-1:]})
    latlong = np.array([[p[0][1], p[0][0]] for p in df['POLYLINE'] if len(p) > 0])

    # cut off long distance trips
    lat_low, lat_hgh = np.percentile(latlong[:, 0], [2, 98])
    lon_low, lon_hgh = np.percentile(latlong[:, 1], [2, 98])

    # create image
    bins = 513
    lat_bins = np.linspace(lat_low, lat_hgh, bins)
    lon_bins = np.linspace(lon_low, lon_hgh, bins)
    H2, _, _ = np.histogram2d(latlong[:, 0], latlong[:, 1], bins=(lat_bins, lon_bins))

    img = np.log(H2[::-1, :] + 1)

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Taxi trip end points')
    # plt.savefig("taxi_trip_end_points.png")
    plt.show()


def visualize_heatmap(predefined_range=True, predefined_center=True, train_filepath='./data/Porto/train.csv',
                      save_folder='./'):
    """
    Plot the trips heatmap/density
    :param predefined_center:
    :param predefined_range:
    :param train_filepath:
    :param save_path:
    :return:
    """
    # using all the trajectories would get a heat map code taken from [2]
    # [2] https://www.kaggle.com/code/mcwitt/heatmap
    # slightly different code to plot the trip density (similar as this heatmap) [3]
    # [3] https://www.kaggle.com/code/thomas92/plot-of-trips

    data_df = pd.read_csv(train_filepath, chunksize=10_000, usecols=['POLYLINE'],
                          # iterator=True, # not sure necessary
                          converters={'POLYLINE': lambda x: json.loads(x)})

    # Copy the iterator: (get_chunks does not exist)
    # https://stackoverflow.com/questions/67892077/pandas-iterating-over-chunks-more-than-once
    # chunks1 = data_df.get_chunk()
    # chunks2 = data_df.get_chunk()

    if predefined_range:
        bins = 1000
        lat_min, lat_max = 41.04961, 41.24961
        lon_min, lon_max = -8.71099, -8.51099

        # process data in chunks to avoid using too much memory
        z = np.zeros((bins, bins))

        # chunksize define in read_csv()
        for chunk in data_df:
            # for chunk in chunks1:
            latlon = np.array([(lat, lon)
                               for path in chunk.POLYLINE
                               for lon, lat in path if len(path) > 0])

            z += np.histogram2d(*latlon.T, bins=bins, range=[[lat_min, lat_max], [lon_min, lon_max]])[0]

        log_density = np.log(1 + z)

        plt.imshow(log_density[::-1, :],  # flip vertically
                   # extent=[lat_min, lat_max, lon_min, lon_max])
                   # Unless mistaken the latitudes and longitude were flipped
                   # https://stackoverflow.com/questions/6999621/how-to-use-extent-in-matplotlib-pyplot-imshow
                   extent=[lon_min, lon_max, lat_min, lat_max])

        plt.savefig(f'{save_folder}heatmap.png')
        print(f'saved to {save_folder}heatmap.png')

    if predefined_center:
        # Longitude and latitude coordinates of Porto
        lat_mid = 41.1496100
        lon_mid = -8.6109900
        dv = 0.1  # delta variation

        nrbins = 2000
        hist = np.zeros((nrbins, nrbins))

        # Note: it seems that cannot iterate in chunks again if depleted iterator once already ?
        # so have to read data again ? Doubles the runtime though. Can copy the iterator.
        data_df = pd.read_csv('./data/Porto/train.csv', chunksize=1000, usecols=['POLYLINE'],
                              iterator=True,  # not sure necessary
                              converters={'POLYLINE': lambda x: json.loads(x)})

        # Pandas >= 1.2 chunk with context manager
        # with pd.read_csv(train_filepath, chunksize=1000, iterator=True, usecols=['POLYLINE'],
        # converters={'POLYLINE': lambda x: json.loads(x)}) as reader:
        # for chunk in reader:
        #         process(chunk)

        for chunk in data_df:
            # for chunk in chunks2:
            # Get just the longitude and latitude coordinates for each trip
            latlong = np.array([coord for coords in chunk['POLYLINE'] for coord in coords if len(coords) > 0])

            # Compute the histogram with the longitude and latitude data as a source
            hist_new, _, _ = np.histogram2d(x=latlong[:, 1], y=latlong[:, 0], bins=nrbins,
                                            range=[[lat_mid - dv, lat_mid + dv], [lon_mid - dv, lon_mid + dv]])

            # Add the new counts to the previous counts
            hist = hist + hist_new

        # We consider the counts on a logarithmic scale
        img = np.log(hist[::-1, :] + 1)

        # Plot the counts
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(img, extent=[lon_mid - dv, lon_mid + dv, lat_mid - dv, lat_mid + dv])
        # plt.axis('off')

        plt.savefig(f'{save_folder}trips_density.png')
        print(f'saved to {save_folder}trips_density.png')


def compute_histogram2d(lats, lons, lat_ckpts, lon_ckpts, extend_ckpts=False):
    """
    Compute a 2D histogram
    :param lats: the list of latitudes
    :param lons: the list of longitudes
    :param lat_ckpts: the list of latitude checkpoints used as bins for the histogram
    :param lon_ckpts: the list of longitude checkpoints used as bins for the histogram
    :param extend_ckpts: extend the list of checkpoints so that samples on hist border are not merged in one cell.
    :return: a 2D histogram
    """
    if extend_ckpts:
        # Problem before was that gave the bin edges and ckpts should have been in the middle of the bins
        # and not on the edges
        extended_lat_ckpts, extended_lon_ckpts = extend_checkpoints(lat_ckpts, lon_ckpts)
        hist2d, _, _ = np.histogram2d(lats, lons, bins=(extended_lat_ckpts, extended_lon_ckpts))
    else:
        # n=len(lat_ckpts)=len(lon_ckpts) then have n-1 cells and n edges for hist2d, this updates the counts on
        hist2d, _, _ = np.histogram2d(lats, lons, bins=(
        lat_ckpts, lon_ckpts))  # range=[[lat_ckpts[1], lat_ckpts[-2]],[lon_ckpts[1], lon_ckpts[-2]]]

    return hist2d


def extend_checkpoints(lats_ckpt, lons_ckpts):
    # The step can also be returned by np.linspace if retstep=true
    lat_delta, lon_delta = lats_ckpt[1] - lats_ckpt[0], lons_ckpts[1] - lons_ckpts[0]
    # Lats
    extended_lat_ckpts = [lats_ckpt[0] - lat_delta]
    extended_lat_ckpts.extend(lats_ckpt)
    extended_lat_ckpts.append(lats_ckpt[-1] + lat_delta)
    # Lons
    extended_lon_ckpts = [lons_ckpts[0] - lon_delta]
    extended_lon_ckpts.extend(lons_ckpts)
    extended_lon_ckpts.append(lons_ckpts[-1] + lon_delta)
    return extended_lat_ckpts, extended_lon_ckpts


class LogPlusOneNorm(matplotlib.colors.Normalize):
    """
    Log(x+1) scaling
    """

    # https://matplotlib.org/3.2.2/gallery/userdemo/colormap_normalizations_custom.html
    # can check base norm in https://github.com/matplotlib/matplotlib/blob/v3.5.2/lib/matplotlib/colors.py#L1608-L1618
    def __init__(self, vmin=None, vmax=None, clip=False):
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        self._transform = lambda x: np.log(x + 1)

    def __call__(self, value, clip=None):
        # ignoring masked values and all kinds of edge cases
        if self.clip:
            value = np.clip(value, self.vmin, self.vmax)
        t_value = self._transform(value)  # np.log(value+1)
        t_vmin, t_vmax = self._transform(self.vmin), self._transform(
            self.vmax)  # math.log(self.vmin + 1), math.log(self.vmax + 1)
        t_value -= t_vmin
        t_value /= (t_vmax - t_vmin)
        return t_value


def plot_hist2d(lats, lons, lat_ckpts, lon_ckpts, title=None, savepath=None, cmap=None, vmin=None, vmax=None,
                extend_ckpts=False):
    """
    Plot hist2D directly without first computing it with numpy.
    Advantage being that get vectorized image (contrary to imshow)
    :param lats: the list of latitudes
    :param lons: the list of longitudes
    :param lat_ckpts: the list of latitude checkpoints used as bins for the histogram
    :param lon_ckpts: the list of longitude checkpoints used as bins for the histogram
    :param title:
    :param savepath:
    :param extend_ckpts: if extend ckpts list left and right. Avoid merging of cells on border of grid in hist2d
    :return:
    """
    if extend_ckpts:
        lon_ckpts, lat_ckpts = extend_checkpoints(lat_ckpts, lon_ckpts)
    # matplotlib.colors.LogNorm() uses numpy masked array to remove the value <= 0 from log computation
    # so some colors are different ie white (no color) when value was 0
    # and then color map scale is changed as normalized a different range of value
    # https://github.com/matplotlib/matplotlib/blob/v3.5.2/lib/matplotlib/colors.py#L1608-L1618
    # plt.hist2d(lons, lats, bins=(lon_ckpts, lat_ckpts), norm=matplotlib.colors.LogNorm())
    # With this custom norm plot should be identical to what show_histogram2d() outputs
    # Can usse figsize (e.g., figsize=(5,5)) (in inches) to set the aspect ratio to 1 to have square cells and not rectangles
    # plt.hist2d(lons, lats, bins=(lon_ckpts, lat_ckpts), norm=LogPlusOneNorm())
    # other way to set aspect ratio with axis.set_aspect()
    fig, ax = plt.subplots()
    if cmap is None:
        ax.hist2d(lons, lats, bins=(lon_ckpts, lat_ckpts), norm=LogPlusOneNorm())
    else:
        ax.hist2d(lons, lats, bins=(lon_ckpts, lat_ckpts), norm=LogPlusOneNorm(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_aspect(1)

    if title is not None:
        plt.title(title)
    plt.savefig(savepath)  # bbox_inches='tight'
    plt.savefig(f'{savepath[:-4]}.svg')
    if LATEX_SAVE:
        plt.savefig(f'{savepath[:-4]}.pdf')
        plt.savefig(f'{savepath[:-4]}.pgf')
    plt.close()


def show_histogram2d(hist2d, extent, title='Hist2D', savepath=None):
    """
    Plot the 2D histogram
    :param hist2d: 2D histogram
    :param extent: extent for plt.imshow() [lon_min, lon_max, lat_min, lat_max]
    :return:
    """
    img = np.log(hist2d[::-1, :] + 1)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(img, extent=extent)  # would be standard image not vectorized
    # plt.axis('off')
    plt.title(title)
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
        if LATEX_SAVE:
            plt.savefig(f'{savepath[:-4]}.pdf')
            plt.savefig(f'{savepath[:-4]}.pgf')
        plt.close()
    else:
        plt.show()


def show_hist2d_target_recovered(hist2d_target, hist2d_recovered, extent, title=None, save_fullpath=None):
    img_target = np.log(hist2d_target[::-1, :] + 1)
    img_recovered = np.log(hist2d_recovered[::-1, :] + 1)

    # For the colormap values could check
    # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    # print(f'DEV:hist2d target:\n{img_recovered}')
    fig, (ax1, ax2) = plt.subplots(1, 2)  # constrained_layout=True, figsize=(7, 3.5)
    ax1.imshow(img_target, extent=extent, cmap='binary')
    ax1.set_title('Target')
    # Specify vmin and vmax since it can happend that the max value is not present
    # (no minhash value equal between signatures), but want to keep consistency between the colors
    # By default the color map is used on value in the range [vmin, vmax] after normalization ?
    # log(3) comes from hist2d value being in range [0,2] but mapped to log(x+1)
    # print(f'DEV:hist2d recovered:\n{img_recovered}')
    ax2.imshow(img_recovered, extent=extent, vmin=0, vmax=math.log(3), cmap='binary')
    ax2.set_title('Recovered')
    if title is not None:
        fig.suptitle(title)  # fontsize=20

    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html
    # Subplot mosaic (works with newer version of matplotlib?)
    # fig, ax = plt.subplot_mosaic([
    #     ['target', 'recovered']
    # ], ) # figsize=(7, 3.5)
    # fig.suptitle(title, fontsize=20)
    #
    # ax['target'].imshow(img_target, extent=extent)
    # ax['target'].set_title('Target')
    # ax['recovered'].imshow(img_recovered, extent=extent)
    # ax['recovered'].set_title('Recovered')

    # set the spacing between subplots
    # Not optimal
    # plt.subplots_adjust(left=0.1,
    #                     bottom=0.1,
    #                     right=0.9,
    #                     top=0.9,
    #                     wspace=0.4,
    #                     hspace=0.4)

    fig.tight_layout()
    # interactive:
    # plt.subplot_tool()

    if save_fullpath is not None:
        # Since save_fullpath is a file create parent directories if necessary
        Path(save_fullpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fullpath)  # with the updated rcparams for latex, also save raw plot as png
        if LATEX_SAVE:  # Note this line may need global variable LATEX_SAVE if run without pycharm python console
            plt.savefig(f'{save_fullpath[:-4]}.pdf')
            plt.savefig(f'{save_fullpath[:-4]}.pgf')
        # Close the fig so that it frees memory ?
        # Due to warning RuntimeWarning: Figures created through the pyplot interface (`matplotlib.pyplot.figure`)
        # are retained until explicitly closed and may consume too much memory.
        # (To control this warning, see the rcParam `figure.max_open_warning`).
        plt.close(fig)  # if do it after show it will close the prompt ?
    else:
        plt.show()


def plot_hist2d_target_recovered(lats_target, lons_target, lats_recovered, lons_recovered, lat_ckpts, lon_ckpts,
                                 title=None, save_fullpath=None, extend_ckpts=True):
    """ same as show_hist2d_target_recovered() and compute compute_histogram2d() combined in one so that
    do not use plt.imshow() and its non vectorized images.
    """
    if extend_ckpts:
        lon_ckpts, lat_ckpts = extend_checkpoints(lat_ckpts, lon_ckpts)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist2d(lons_target, lats_target, bins=(lon_ckpts, lat_ckpts), norm=LogPlusOneNorm(), cmap='binary')
    ax1.set_aspect(1)
    ax1.set_title('Target')

    # Here since vmax is transformed by LogPlusOneNorm() need to put 2 and not math.log(3) ?
    ax2.hist2d(lons_recovered, lats_recovered, bins=(lon_ckpts, lat_ckpts), norm=LogPlusOneNorm(),
               vmin=0, vmax=2, cmap='binary')
    ax2.set_aspect(1)
    ax2.set_title('Recovered')

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    if save_fullpath is not None:
        Path(save_fullpath).parent.mkdir(parents=True, exist_ok=True)
        # Note pdflatex pgfplots: `TeX capacity exceeded, sorry [main memory size=3000000].` error
        plt.savefig(save_fullpath)
        if LATEX_SAVE:  # Note this line may need global variable LATEX_SAVE if run without pycharm python console
            plt.savefig(f'{save_fullpath[:-4]}.pdf')
            plt.savefig(f'{save_fullpath[:-4]}.pgf')
        plt.close(fig)
    else:
        plt.show()


def get_trajectories(start_id, end_id, offset=0, filepath='./data/Porto/train.csv'):
    """
    Obtain the trajectories from Porto taxi dataset
    :param start_id: starting index for the trajectory to include
    :param end_id: ending index for the trajectory to include
    :param offset: offset for the indices
    :param filepath: filepath to the trajectory data
    :return: a list of trajectory (represented as a list of coordinates, and a coordinate is a list: [lat, lon])
    """
    print(f'reading porto taxi dataset...')

    # can check visualize_heatmap() for other read_csv params
    df = pd.read_csv(filepath, converters={'POLYLINE': lambda x: json.loads(x)})
    # drop unused columns (for train TRIP_ID seems to be TIMESTAMP + TAXI_ID (where + is concatenation)
    df.drop(inplace=True, columns=['CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'DAY_TYPE', 'MISSING_DATA'])
    # trajectory_s = df.loc[:, 'POLYLINE'].head(first_n)  # [['POLYLINE']] if want it to return a dataframe [] returns serie
    # Note: could shuffle dataframe
    trajectory_s = df.loc[offset + start_id:offset + end_id, 'POLYLINE']

    # not best to iterate over pandas dataframe, better to vectorize etc. list comprehension have some acceleration ?
    trajectory_list = [polyline for polyline in trajectory_s]
    # if too slow could save selected trajectories and then restore from save file
    return trajectory_list


def trajectory_generator(split_count, split_size=30_000, filepath='./data/Porto/train.csv'):
    """

    :param split_count: the number of data split to include
    :param split_size: size of the data split (default 30_000)
    :param filepath: filepath to the trajectory data
    :return: lazy iterator
    """
    trajectory_list = get_trajectories(0, split_count * split_size, filepath=filepath)
    print(f'trajectory_list [0, {len(trajectory_list)}[')
    # Lazy iterator, avoid having too much content in memory
    for i in range(split_count):
        cur_offset = i * split_size
        print(f'trajectory split [{cur_offset},{cur_offset + split_size}[')
        yield trajectory_list[cur_offset:cur_offset + split_size]


def binary_search_closest(sorted_data, target_value):
    """
    Find the value closest to target using binary search
    :param sorted_data: sorted iterable
    :param target_value: target value
    :return: closest value to target
    """
    # https://stackoverflow.com/questions/23681948/get-index-of-closest-value-with-binary-search
    low, high = 0, len(sorted_data) - 1
    best_ind = low
    while low <= high:
        mid = low + (high - low) // 2
        if sorted_data[mid] < target_value:
            low = mid + 1
        elif sorted_data[mid] > target_value:
            high = mid - 1
        else:
            best_ind = mid
            break
        # check if sorted_data[mid] is closer to target_value than sorted_data[best_ind]
        if abs(sorted_data[mid] - target_value) < abs(
                sorted_data[best_ind] - target_value):  # may need <= if non unique values
            best_ind = mid
    return best_ind


def update_ckpt_sig(ckpt_sig, mobile_entity_sig):
    """
    Update according to minHashTrajectory paper. A checkpoint updates its signature according by keeping the minimum coordinate
    of every mobile entity signature it encounters in a time period.
    :param ckpt_sig: the current checkpoint signature (as an iterable like list)
    :param mobile_entity_sig: the current mobile entity passing by the checkpoint and sending its signature (as an iterable like list)
    :return: the updated checkpoint signature
    """
    # Want the element-wise minimum of the two iterable (list)
    updated_ckpt_sig = np.minimum(ckpt_sig, mobile_entity_sig).tolist()
    return updated_ckpt_sig


def generate_mobile_entities_sig(hash_list, entity_count, minhash_inset_sample_range, mode='random'):
    """
    :param hash_list: list of hash function used for minhash
    :param entity_count: number of signature to generate, one per mobile entity
    :param minhash_inset_sample_range: the range of integer values to use to sample the minhash input set from
    :param mode: one of 'random'
    :return: the list of minhash signature for each entity
    """
    # The number of hash function determine signature length
    sig_len = len(hash_list)
    if mode == 'random':
        # Use the entity id as seed to generate a set
        vehicle_signatures = []

        # Need to define that parameter
        input_set_size = max(sig_len, int(math.sqrt(entity_count)))

        for entity_id in range(1, entity_count + 1):
            # note that can have collisions here since two different entities can have same set elements mapping to same hashed value
            #  meaning that when entities signature has value equal to antenna signature cannot be certain this vehicle passed here
            # minhash_input_set = generate_target_set(input_set_size, sample_range=minhash_inset_sample_range, seed=entity_id)
            # use singleton set
            minhash_input_set = {entity_id}
            # compute minhash signature:
            c_sig = compute_hashfunc_minhash_signature(minhash_input_set, hash_list)
            vehicle_signatures.append(c_sig)

        return vehicle_signatures


def get_checkpoint_index(lat, lon, lat_check, lon_check):
    """
    Compute the checkpointID for the current coordinates
    :param lat: the latitude of the coordinate under consideration
    :param lon: the longitude
    :param lat_check: the sorted ndarray of checkpoints' latitudes
    :param lon_check: the sorted ndarray of checkpoints' longitude
    :return: the checkpoint index
    """
    lat_id = binary_search_closest(lat_check, lat)
    lon_id = binary_search_closest(lon_check, lon)
    # Check proximity to checkpoints:
    # if traj_id % 10 == 0:
    # print(f'{lat_id} {lat_check[lat_id]} =~= {lat}')
    # print(f'{lon_id} {lon_check[lon_id]} =~= {lon}')
    # Only works for grid ie if number of checkpoints for latitude and longitudes
    # ckpt_id = lat_id * lon_check.size + lon_id # size for numpy array
    ckpt_id = lat_id * len(lon_check) + lon_id  # for ndarray len is only for first dimension

    return ckpt_id


def get_checkpoint_coordinates(ckpt_id, lat_ckpt, lon_ckpt):
    """
    recover coordinates from index
    :param ckpt_id: index of checkpoints
    :param lat_ckpt: latitudes of checkpoints
    :param lon_ckpt: longitudes of checkpoints
    :return: coordinate tuple (lat, lon)
    """
    # Works if 0 <= lon_id, lat_id < len(lon_ckpt)
    # lon_id = ckpt_id % lon_ckpt.size
    lon_id = ckpt_id % len(lon_ckpt)  # len() same as .size only if one dimensional (ie len of first dim only)
    lat_id = ckpt_id // len(lon_ckpt)
    return (lat_ckpt[lat_id], lon_ckpt[lon_id])  # , (lat_id, lon_id)


def compare_minhash_signatures(ckpt_sig, vehicle_sig):
    """

    :param ckpt_sig: minhash signature of the checkpoint
    :param vehicle_sig: mobile entity signature
    :return: boolean, int tuple, if vehicle can have cross checkpoint, and if at least one value of signature is equal
    (returns the count of such occurence, the int interpreted as boolean, false for 0 true otherwise)
    """
    if len(ckpt_sig) != len(vehicle_sig):
        raise Exception('signature should be of same length')
    sig_i_equal_count = 0
    for i in range(len(ckpt_sig)):
        if vehicle_sig[i] < ckpt_sig[i]:
            # if one element of the vehicle signature is smaller than the checkpoint minhash signature
            # by construction of checkpoint signature we have a contradiction, if the vehicle would have passed by
            # that checkpoint, the checkpoint signature would have been updated and be less or equal to vehicle sig
            return False, False
        elif vehicle_sig[i] == ckpt_sig[i]:
            sig_i_equal_count += 1
            # return True, True
        # else: # can happen that

    # No contradiction with checkpoint signature but no equality of value
    # return True, False
    return True, sig_i_equal_count


def get_lat_lon_for_traj(ckpt_list, ckpt_whp_list, lat_ckpts, lon_ckpts, keep_cardinality=False):
    """

    :param ckpt_list: list of checkpoint of trajectory, can be a set if want binary color map (keep_cardinality=true)
    :param ckpt_whp_list: if checkpoint had equality with vehicle (meaning vehicle passed by ckpt with high probabily)
    :param keep_cardinality: if use set for binary color map hist2D later or use list directly to keep count of ckpts
    :return:
    """
    latitudes = []
    longitudes = []
    ckpt_visited_set = set()  # problem checkpoints are different anyway it seems may need to clip histogram for value outside bins ?
    for ckpt_id in ckpt_list:
        if not keep_cardinality:
            # Ensure uniqueness of ckpt (since ckpt outside the grid map are clipped to the borders)
            # cannot do it on latitudes and longitudes since can have same lat but different lon
            if ckpt_id in ckpt_visited_set:
                continue  # go to next iteration
            else:
                ckpt_visited_set.add(ckpt_id)
        lat, lon = get_checkpoint_coordinates(ckpt_id, lat_ckpts, lon_ckpts)
        latitudes.append(lat)
        longitudes.append(lon)

    # For the colormap values could check
    # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    for ckpt_id in ckpt_whp_list:
        lat, lon = get_checkpoint_coordinates(ckpt_id, lat_ckpts, lon_ckpts)
        # give a higher count to histogram by adding the ckpt coordinate more time
        # if isinstance(ckpt_list, set):
        if keep_cardinality:
            latitudes.append(lat)
            longitudes.append(lon)
        else:
            latitudes.extend([lat] * 20)
            longitudes.extend([lon] * 20)

    return latitudes, longitudes


def save_figure(cur_value, threshold_value_list, operator_list, counter_name_list, counter):
    """
    Check if save the current figure
    :param cur_value: current value to compare
    :param threshold_value_list: list of len 2 for the threshold value
    :param operator_list: list of len 2 for the operator
    :param counter_name_list: list of len 3 for the keys for the counter
    :param counter: the counter collection
    :return:
    """
    compute_hist2d = False
    if operator_list[0](cur_value, threshold_value_list[0]):  # only care about good ratios ?
        if counter[counter_name_list[0]] > 0:
            counter[counter_name_list[0]] -= 1
            compute_hist2d = True
    elif operator_list[1](cur_value, threshold_value_list[1]):
        if counter[counter_name_list[1]] > 0:
            counter[counter_name_list[1]] -= 1
            compute_hist2d = True
    else:  # 0.25 < tr_ratio < 0.75 (valid python condition)
        if counter[counter_name_list[2]] > 0:
            counter[counter_name_list[2]] -= 1
            compute_hist2d = True
    return compute_hist2d


if __name__ == '__main__':
    # visualize_endpoints()
    # visualize_heatmap()

    use_dev_test_param = False
    LATEX_SAVE = False
    imshow_low_res = False
    # with lon_check (middle of bin) and lon_hist_bins_edges (bin edges) separation no need to extend hist anymore
    extend_hist2d_ckpts = False
    # Note would most likely fail if set to False, due to reuse of latitudes, longitudes derived variable
    compute_ckpt_stats = True
    use_cryptographic_hash = False

    # To save for latex.
    # https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
    if LATEX_SAVE:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({  # Reset the matplotlib params at end of file
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    temp_save_folder_path = f'./logs/run_{datetime.now():%d-%m-%Y_at_%Hh%Mm%Ss}/'
    Path(temp_save_folder_path).mkdir(parents=True, exist_ok=True)
    print(f'saving console output to: {temp_save_folder_path}')
    # Redirect console output to file
    # Before piped it to console and file with loggers see utils
    # tqdm progress bar are not saved in file but printed in console
    sys.stdout = open(f'{temp_save_folder_path}/console_output.txt', 'w')

    if use_dev_test_param:
        # parameters for smaller size tests
        n, k = 3000, 20
        lat_ckpt_cnt, lon_ckpt_cnt = 10, 10
        split_n = 1
        np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=160)
    else:
        n = 30000  # Trajectory counts
        lat_ckpt_cnt, lon_ckpt_cnt = 88, 88  # if want around 8000 checkpoints 89*89=7921, 88*88=7744 approx. their 7714 in table 1
        k = 200  # Hash counts
        split_n = 5
    m = lat_ckpt_cnt * lon_ckpt_cnt  # checkpoint counts

    print(f'Parameters: traj: {n} ckpt: {lat_ckpt_cnt, lon_ckpt_cnt}, hash: {k}')

    # Statistics
    print(f'Use cryptographic hash: {use_cryptographic_hash}, compute ckpt stats: {compute_ckpt_stats}')

    # Requirements for mobile entity signatures
    # Generate hash functions
    if use_cryptographic_hash:
        # hash list for minhash
        mh_hash_list = generate_cryptographic_hash_list(k)
        # Here would limit the sample range for the set elements that will go through hash function
        # for cryptographic hash it is not important
        MAX_POSSIBLE_VALUE = 1 << 256
    else:
        # easier for inspection of signature with smaller numbers
        # Check minhash.py for choice of primes
        prime_p = 4294967311  # 20 < 23; 1000 < ; 10000 < 10007; 2^32 < 4294967311, (1<<128)-160 < 2^128 - 159 (prime)
        # No need to generate above the maximum value since operation modulo p
        MAX_POSSIBLE_VALUE = prime_p - 1
        print(f'sampling random coefficients')
        r_coeffs, c_coeffs = sample_random_coefficients(k, 0, MAX_POSSIBLE_VALUE, seed=None)

        print(f'creating hash functions')
        mh_hash_list = hash_list_from_coeffs(r_coeffs, c_coeffs, prime_p)

    # Mobile entity signatures
    # According to paper vehicle (mobile entity) sends its signature to checkpoints
    # and checkpoints udpates wherever smaller

    # n is number of trajectory but for now assume only unique entities did a trajectory
    mobile_entities_sig = generate_mobile_entities_sig(mh_hash_list, n, (0, MAX_POSSIBLE_VALUE))

    # if get ram exhaustion can process data in chunks as done in visualize_heatmap()
    # trajectory_list = get_trajectories(n)
    # trajectory_list = get_trajectories(0, n)
    split_id = 0
    ckpt_set_avg_len_target_list = []
    ckpt_set_len_ratio_total_ckpt_avg_list_target = []  # same as above but directly aggregate ratios
    ckpt_set_avg_len_recovered_list = []
    ckpt_set_len_ratio_total_ckpt_avg_list_recovered = []  # same as above but directly aggregate ratios
    for trajectory_list in trajectory_generator(split_count=split_n, split_size=n):
        if compute_ckpt_stats:
            ## List all latitudes and longitudes
            # Note porto dataset uses format [lon, lat] while standard would be [lat, lon]
            latitudes = [latlon_2list[1] for polyline in trajectory_list for latlon_2list in polyline]
            longitudes = [latlon_2list[0] for polyline in trajectory_list for latlon_2list in polyline]
            lat_min, lat_max = min(latitudes), max(latitudes)
            lon_min, lon_max = min(longitudes), max(longitudes)
            # Too long to fit in the console print default buffer size
            # print(f'[Lat|Long]-itudes:\n{sorted(latitudes)}\n{sorted(longitudes)}')
            print(f'Lat: {lat_min, lat_max} Lon: {lon_min, lon_max}')

            # Lat and lon checkpoint count
            # lat_check = np.linspace(lat_min, lat_max, num=lat_ckpt_cnt)
            # lon_check = np.linspace(lon_min, lon_max, num=lon_ckpt_cnt)

            # cut off long distance trips
            # Try to remove outliers to ensure better checkpoints spread
            # outlier would be clipped to boundaries of lat and lon checkpoints
            lat_low, lat_hgh = np.percentile(latitudes, [2, 98])
            lon_low, lon_hgh = np.percentile(longitudes, [2, 98])
            lat_check, lat_check_step = np.linspace(lat_low, lat_hgh, num=lat_ckpt_cnt, retstep=True)
            lon_check, lon_check_step = np.linspace(lon_low, lon_hgh, num=lon_ckpt_cnt, retstep=True)
            # For hist want the lat and lon check to be in the middle of the hist2d edges and not to be the edges
            # print(f'DEBUG: ckpts step {lat_check_step, lon_check_step}')
            # Want lat_check, lon_check to be in the middle of each hist2d cell defined by 2 consecutive bin edges
            lat_hist_bins_edges, lat_hist_bins_step = np.linspace(lat_low - lat_check_step / 2,
                                                                  lat_hgh + lat_check_step / 2, num=lat_ckpt_cnt + 1,
                                                                  retstep=True)
            lon_hist_bins_edges, lon_hist_bins_step = np.linspace(lon_low - lon_check_step / 2,
                                                                  lon_hgh + lon_check_step / 2, num=lon_ckpt_cnt + 1,
                                                                  retstep=True)
            if lat_check_step != lat_hist_bins_step or lon_check_step != lon_hist_bins_step:
                print(f'The step should be approximately the same {lat_check_step} =?= {lat_hist_bins_step}')
                print(f'The step should be approximately the same {lon_check_step} =?= {lon_hist_bins_step}')

            # Show 2D histogram of selected checkpoint and trajectories passing by ?
            # before used lon_check instead of lon_hist_bins_edges, changed since want checkpoint in the middle of the bin.
            hist2d = compute_histogram2d(latitudes, longitudes, lat_hist_bins_edges, lon_hist_bins_edges,
                                         extend_ckpts=extend_hist2d_ckpts)
            visited_ckpt_count = np.count_nonzero(hist2d)
            show_histogram2d(hist2d, [lon_low, lon_hgh, lat_low, lat_hgh],
                             title='Selected trajectory checkpoint 2D histogram',
                             savepath=f'{temp_save_folder_path}split_{split_id}/all_trajectories_(visited-ckpt={visited_ckpt_count}).png')
            # Changed  lat_check, lon_check to lat_hist_bins_edges, lon_hist_bins_edges, since want checkpoint in middle of 2 bin edges
            plot_hist2d(latitudes, longitudes, lat_hist_bins_edges, lon_hist_bins_edges,
                        savepath=f'{temp_save_folder_path}split_{split_id}/all_trajectories_(visited-ckpt={visited_ckpt_count})_bis.png',
                        extend_ckpts=extend_hist2d_ckpts)

            # Cartesian product https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
            # Do not need to represent cartesian product, can use index (x,y) or combined as x*len(y) + y or similar
            # checkpoints = list(range(m))

            # Count checkpoints updates for each trajectory
            ckpt_update_count = [[0] * m for _ in range(n)]
            ckpt_update_count_1pt = [0] * m  # 1 checkpoint update per trajectories

        else:  # Code would probably not work anymore if this condition is true
            # np.arange(-8.7, -8.5, step=0.1) # here specify steps
            # Lat and lon checkpoint count (if have predefined limits for porto city under consideration
            lat_check = np.linspace(-8.689, -8.575, num=lat_ckpt_cnt)
            lon_check = np.linspace(41.134, 41.181, num=lon_ckpt_cnt)

        print(f'Computing checkpoints signatures...')
        # Init checkpoint signature to infinity vectors
        checkpoints_sig = [[math.inf] * k for _ in range(m)]
        # Trajectories in checkpoints
        trajectory_ckpt_list = []
        # Update checkpoints for trajectory
        for traj_id, trajectory in enumerate(tqdm(trajectory_list, desc='update checkpoints')):
            # Maintain a set of already updated checkpoints (avoid reupdating one for same trajectory id
            checkpoints_updated = set()
            trajectory_ckpts = []
            # Note Porto dataset uses format [lon, lat] while standard is [lat, lon]
            # also some trajectory may be empty
            for [lon, lat] in trajectory:
                cur_ckpt = get_checkpoint_index(lat, lon, lat_check, lon_check)
                trajectory_ckpts.append(cur_ckpt)

                if cur_ckpt not in checkpoints_updated:
                    checkpoints_updated.add(cur_ckpt)
                    # update checkpoint signature (elementwise minimum)
                    checkpoints_sig[cur_ckpt] = update_ckpt_sig(checkpoints_sig[cur_ckpt], mobile_entities_sig[traj_id])
                    if compute_ckpt_stats:
                        ckpt_update_count_1pt[cur_ckpt] += 1

                if compute_ckpt_stats:
                    ckpt_update_count[traj_id][cur_ckpt] += 1

            # Note since in dataset row 763 (762 if 0-based indexing) has an empty trajectory
            # though we need to have the same indexing so keep the empty trajectory ?
            # if len(trajectory_ckpts) > 0:

            trajectory_ckpt_list.append(trajectory_ckpts)

        if compute_ckpt_stats:
            ckpt_update_count_1pt_non_0 = [(i, ckpt) for i, ckpt in enumerate(ckpt_update_count_1pt) if ckpt != 0]
            print(ckpt_update_count_1pt_non_0)
            # better inspection with pycharm SciView if use Numpy:
            ckpt_update_count = np.array(ckpt_update_count)
            ckpt_update_count_1pt = np.array(ckpt_update_count_1pt)
            checkpoints_sig_np = np.array(checkpoints_sig)

        max_plot_to_save = {'tr_ratio>0.75': 10, 'tr_ratio<0.25': 10, '0.25<tr_ratio<0.75': 10,
                            't_ckpt_cnt>2000': 10, 't_ckpt_cnt<200': 10, '200<t_ckpt_cnt<2000': 10}
        saved_trajectory_plot_counter = Counter(max_plot_to_save)
        # reconstruct a superset of the target trajectory
        target_recovered_ratios = []
        target_recovered_mh_ratios = []
        trajectory_ckpt_set_len = []
        recovered_traj_ckpt_set_len = []
        recovered_mh_eq_ckpt_set_len = []
        # Iterate over all target trajectory (to recover target vehicle signatures etc)
        for traj_id in trange(len(trajectory_ckpt_list),
                              desc=f'Reconstruct checkpoint trajectories'):  # before len(trajectory_list) but for now ensured same length with trajectory_ckpt_list
            potential_ckpt_traj_list = []
            potential_ckpt_traj_set = set()
            mh_value_eq_ckpt_traj_list = []  # may be needed for hist2D
            mh_value_eq_ckpt_traj_set = set()  # was casting the list as set multiple time
            # Check for all the checkpoints signature if vehichle could have passed by
            for ckpt_id, ckpt_sig in enumerate(checkpoints_sig):
                # now mh_value_eq_count returns number of equal signature entries but nothing done with it currently (though could as post processing)
                #  the more value collide the more probable it is for checkpoint to be in target trajectory
                valid_ckpt, mh_value_eq_count = compare_minhash_signatures(ckpt_sig, mobile_entities_sig[traj_id])
                if valid_ckpt:
                    potential_ckpt_traj_list.append(
                        ckpt_id)  # possible checkpoint repetitions, though should not by construction
                    potential_ckpt_traj_set.add(ckpt_id)  # only care about unique checkpoints
                if mh_value_eq_count:  # note changed this value now it returns an int (interpreted as boolean)
                    mh_value_eq_ckpt_traj_list.append(ckpt_id)
                    mh_value_eq_ckpt_traj_set.add(ckpt_id)

            # Current target trajectory as a set of checkpoints
            target_ckpt_traj_set = set(trajectory_ckpt_list[traj_id])
            trajectory_ckpt_set_len.append(len(target_ckpt_traj_set))
            # Accumulate current recovered trajectory set of checkpoint count
            recovered_traj_ckpt_set_len.append(len(potential_ckpt_traj_set))
            # Accumulate for the collision in signatures (equality of at least one element)
            recovered_mh_eq_ckpt_set_len.append(len(mh_value_eq_ckpt_traj_set))

            # Sanity check:
            if not target_ckpt_traj_set.issubset(potential_ckpt_traj_set):
                raise Exception(
                    f'Target set should be included in potential set: {target_ckpt_traj_set} in {potential_ckpt_traj_set} diff: {target_ckpt_traj_set - potential_ckpt_traj_set}')
            for value in mh_value_eq_ckpt_traj_set:
                if value not in target_ckpt_traj_set:
                    print(
                        f'{Fore.RED}ATTENTION collision, found a checkpoint that had an equality in at least one signature value but this ckpt is not part of target trajectory')

            # Ratio between number of unique (hence set) checkpoints in target trajectory and recovered superset
            if len(potential_ckpt_traj_set) != 0:
                tr_ratio = len(target_ckpt_traj_set) / len(potential_ckpt_traj_set)
                target_recovered_ratios.append(tr_ratio)
            else:
                print(
                    f'{Fore.RED}division by 0 (potential ckpt traj {traj_id} is empty) {len(target_ckpt_traj_set)} / {len(potential_ckpt_traj_set)}')
                tr_ratio = math.inf
            if len(target_ckpt_traj_set) != 0:
                tr_mh_ratio = len(mh_value_eq_ckpt_traj_set) / len(target_ckpt_traj_set)  # minhash equality
                target_recovered_mh_ratios.append(tr_mh_ratio)
            else:
                print(
                    f'{Fore.RED}division by 0 (trajectory_ckpt_list[{traj_id}] is empty) {len(mh_value_eq_ckpt_traj_set)} / {len(target_ckpt_traj_set)}')
                tr_mh_ratio = math.inf
            # First can have duplicate so transform into set, by construction the two other lists should not have duplicate
            # print(f'counts: target {len(target_ckpt_traj_set)} recovered: {len(potential_ckpt_traj)} whp {len(mh_value_eq_ckpt_traj)} t/r={tr_ratio}, whp/t={tr_mh_ratio}')
            # print(f'{traj_id}\ntarget {trajectory_ckpt_list[traj_id]}\nset {target_ckpt_traj_set}\npotential {potential_ckpt_traj}\nhigh confidence {mh_value_eq_ckpt_traj}')

            if compute_ckpt_stats:
                # Try to plot histogram2d of target trajectory vs recovered ?
                # expensive to save every figure so only save some examples of interesting behavior depending on filtered parameters
                # note lazy evaluation on or so if first condition is true wont evaluate the second and wont decrease counter
                compute_hist2d = \
                    save_figure(tr_ratio, [0.25, 0.75], [operator.lt, operator.gt],
                                ['tr_ratio<0.25', 'tr_ratio>0.75', '0.25<tr_ratio<0.75'], saved_trajectory_plot_counter) \
                    or \
                    save_figure(len(potential_ckpt_traj_set), [200, 800 + 3 * 400], [operator.lt, operator.gt],
                                ['t_ckpt_cnt<200', 't_ckpt_cnt>2000', '200<t_ckpt_cnt<2000'],
                                saved_trajectory_plot_counter)

                # Just saving 60 target in 3 format double total runtime
                if compute_hist2d:
                    # Note with pycharm scientific mode may be too much for the plot visualizer
                    # Target:
                    # Use list as input (color change with number of appearances of checkpoint)
                    # Note: need to adapt compute_histogram2d to support again a different colormap (eg not binary and no vmax=2?)
                    # target_lats, target_lons = get_lat_lon_for_traj(trajectory_ckpt_list[traj_id], [], lat_check, lon_check)
                    # If want set as input (histogram then does not try to color by the number of time checkpoints appear in trajectory)
                    target_lats, target_lons = get_lat_lon_for_traj(target_ckpt_traj_set, [], lat_check, lon_check,
                                                                    keep_cardinality=False)
                    traj_lats, traj_lons = get_lat_lon_for_traj(potential_ckpt_traj_set, mh_value_eq_ckpt_traj_set,
                                                                lat_check, lon_check)
                    save_hist_path = f'{temp_save_folder_path}split_{split_id}/Trajectory-{traj_id}+{split_id * n}' \
                                     f'_len_{len(target_ckpt_traj_set)}_{len(potential_ckpt_traj_set)}_{len(mh_value_eq_ckpt_traj_set)}-' \
                                     f'trr={tr_ratio:.2f}_mhq={tr_mh_ratio:.2f}.png'
                    if use_dev_test_param:
                        # Do both:
                        # imshow low res
                        # changed lat_check, lon_check to lat_hist_bins_edges, lon_hist_bins_edges
                        target_hist2d = compute_histogram2d(target_lats, target_lons, lat_hist_bins_edges,
                                                            lon_hist_bins_edges, extend_ckpts=extend_hist2d_ckpts)
                        traj_hist2d = compute_histogram2d(traj_lats, traj_lons, lat_hist_bins_edges,
                                                          lon_hist_bins_edges, extend_ckpts=extend_hist2d_ckpts)
                        # print(f'DEV:{traj_id}')
                        show_hist2d_target_recovered(target_hist2d, traj_hist2d, [lon_low, lon_hgh, lat_low, lat_hgh],
                                                     # title=f'Traj_{traj_id}+{split_id*n} w. target-recovered ratio={tr_ratio:.3f} mhq={tr_mh_ratio:.3f}',
                                                     save_fullpath=save_hist_path)

                        # plt
                        target_hist_savepath = f'{temp_save_folder_path}split_{split_id}/Trajectory-{traj_id}+{split_id * n}_len_{len(target_ckpt_traj_set)}.png'
                        # changed lat_check, lon_check to lat_hist_bins_edges, lon_hist_bins_edges
                        plot_hist2d(target_lats, target_lons, lat_hist_bins_edges, lon_hist_bins_edges, title=None,
                                    savepath=target_hist_savepath, cmap='binary', extend_ckpts=extend_hist2d_ckpts)
                        recov_hist_savepath = f'{temp_save_folder_path}split_{split_id}/Trajectory-{traj_id}+{split_id * n}' \
                                              f'_len_{len(potential_ckpt_traj_set)}_{len(mh_value_eq_ckpt_traj_set)}-' \
                                              f'trr={tr_ratio:.2f}_mhq={tr_mh_ratio:.2f}.png'
                        plot_hist2d(traj_lats, traj_lons, lat_hist_bins_edges, lon_hist_bins_edges, title=None,
                                    savepath=recov_hist_savepath, vmin=0, vmax=2, cmap='binary',
                                    extend_ckpts=extend_hist2d_ckpts)

                    else:
                        if imshow_low_res:
                            # changed lat_check, lon_check to lat_hist_bins_edges, lon_hist_bins_edges
                            target_hist2d = compute_histogram2d(target_lats, target_lons, lat_hist_bins_edges,
                                                                lon_hist_bins_edges, extend_ckpts=extend_hist2d_ckpts)
                            # show_histogram2d(target_hist2d, [lon_low, lon_hgh, lat_low, lat_hgh], title=f'Traj{traj_id} target')
                            # Recovered:
                            # Using a list
                            # traj_lats, traj_lons = get_lat_lon_for_traj(potential_ckpt_traj_list, mh_value_eq_ckpt_traj_list, lat_check, lon_check)\
                            # If use set (change coloring of histogram due to no repetition of values unless manually repeated)

                            traj_hist2d = compute_histogram2d(traj_lats, traj_lons, lat_hist_bins_edges,
                                                              lon_hist_bins_edges, extend_ckpts=extend_hist2d_ckpts)
                            # show_histogram2d(traj_hist2d, [lon_low, lon_hgh, lat_low, lat_hgh], title=f'Traj{traj_id} recovered')
                            show_hist2d_target_recovered(target_hist2d, traj_hist2d,
                                                         [lon_low, lon_hgh, lat_low, lat_hgh],
                                                         # title=f'Traj_{traj_id}+{split_id*n} w. target-recovered ratio={tr_ratio:.3f} mhq={tr_mh_ratio:.3f}',
                                                         save_fullpath=save_hist_path)
                        else:
                            # Has memory exceeded error
                            # plot_hist2d_target_recovered(target_lats, target_lons, traj_lats, traj_lons, lat_check, lon_check, save_fullpath=save_hist_path)
                            # Split into two files:
                            target_hist_savepath = f'{temp_save_folder_path}split_{split_id}/Trajectory-{traj_id}+{split_id * n}_len_{len(target_ckpt_traj_set)}.png'
                            # changed lat_check, lon_check to lat_hist_bins_edges, lon_hist_bins_edges
                            plot_hist2d(target_lats, target_lons, lat_hist_bins_edges, lon_hist_bins_edges, title=None,
                                        savepath=target_hist_savepath, cmap='binary', extend_ckpts=extend_hist2d_ckpts)
                            recov_hist_savepath = f'{temp_save_folder_path}split_{split_id}/Trajectory-{traj_id}+{split_id * n}' \
                                                  f'_len_{len(potential_ckpt_traj_set)}_{len(mh_value_eq_ckpt_traj_set)}-' \
                                                  f'trr={tr_ratio:.2f}_mhq={tr_mh_ratio:.2f}.png'
                            plot_hist2d(traj_lats, traj_lons, lat_hist_bins_edges, lon_hist_bins_edges, title=None,
                                        savepath=recov_hist_savepath, vmin=0, vmax=2, cmap='binary',
                                        extend_ckpts=extend_hist2d_ckpts)

        print(f'Statistics for split id {split_id}, traj offset={split_id * n}')

        # Ratios between target and recovered checkpoints counts
        tr_ratios_mean, tr_ratios_std = np.mean(target_recovered_ratios), np.std(target_recovered_ratios)
        tr_mh_ratios_mean, tr_mh_ratios_std = np.mean(target_recovered_mh_ratios), np.std(target_recovered_mh_ratios)
        # If remove the zero values
        target_recovered_mh_ratios_wo_0 = [c for c in target_recovered_mh_ratios if c != 0]
        tr_mh_ratios_mean0, tr_mh_ratios_std0 = np.mean(target_recovered_mh_ratios_wo_0), np.std(
            target_recovered_mh_ratios_wo_0)
        print(f'Ratio target recovered ({len(target_recovered_ratios)}): {tr_ratios_mean, tr_ratios_std}')
        print(
            f'Ratio target recovered w minhash eq ({len(target_recovered_mh_ratios)}): {tr_mh_ratios_mean, tr_mh_ratios_std}')
        print(
            f'Ratio target recovered w minhash eq w/o zeros ({len(target_recovered_mh_ratios_wo_0)}): {tr_mh_ratios_mean0, tr_mh_ratios_std0}')

        tr_ratios_quantiles = np.quantile(target_recovered_ratios, [0.25, 0.5, 0.75])
        print(f'tr_ratios quantiles: {tr_ratios_quantiles}')
        score = 0.5
        percentage_below_score = stats.percentileofscore(target_recovered_ratios, score)
        print(
            f'{percentage_below_score}% of values in the target-recovered ratios are below {score} hence {100 - percentage_below_score} are above')
        tr_mh_ratios_quantiles0 = np.quantile(target_recovered_mh_ratios_wo_0, [0.25, 0.5, 0.75])
        print(f'tr_mh_ratios0 quantiles: {tr_mh_ratios_quantiles0}')

        # Average unique checkpoint counts
        trajs_unique_ckpt_count_mean, trajs_unique_ckpt_count_std = np.mean(trajectory_ckpt_set_len), np.std(
            trajectory_ckpt_set_len)
        print(
            f'Average length of target trajectory checkpoint set is: {trajs_unique_ckpt_count_mean} +- {trajs_unique_ckpt_count_std}')
        ckpt_set_avg_len_target_list.append(trajs_unique_ckpt_count_mean)
        # Ratiod by total checkpoint count
        ratiod_trajectory_ckpt_set_len = [cnt / m for cnt in trajectory_ckpt_set_len]
        ratio_uniq_ckpt_cnt_mean, ratio_uniq_ckpt_cnt_std = np.mean(ratiod_trajectory_ckpt_set_len), np.std(
            ratiod_trajectory_ckpt_set_len)
        print(f'Same but ratioed by total checkpoint count: {ratio_uniq_ckpt_cnt_mean} +- {ratio_uniq_ckpt_cnt_std}')
        ckpt_set_len_ratio_total_ckpt_avg_list_target.append(ratio_uniq_ckpt_cnt_mean)

        recov_unique_ckpt_count_mean, recov_unique_ckpt_count_std = np.mean(recovered_traj_ckpt_set_len), np.std(
            recovered_traj_ckpt_set_len)
        print(
            f'Average length of recovered trajectory checkpoint set is: {recov_unique_ckpt_count_mean} +- {recov_unique_ckpt_count_std}')
        ckpt_set_avg_len_recovered_list.append(recov_unique_ckpt_count_mean)
        # Ratioed by total ckpt count
        ratiod_recov_traj_ckpt_set_len = [cnt / m for cnt in recovered_traj_ckpt_set_len]
        ratio_recov_uniq_ckpt_cnt_mean, ratio_recov_uniq_ckpt_cnt_std = np.mean(ratiod_recov_traj_ckpt_set_len), np.std(
            ratiod_recov_traj_ckpt_set_len)
        print(
            f'Same but ratioed by total checkpoint count: {ratio_recov_uniq_ckpt_cnt_mean} +- {ratio_recov_uniq_ckpt_cnt_std}')
        ckpt_set_len_ratio_total_ckpt_avg_list_recovered.append(ratio_recov_uniq_ckpt_cnt_mean)

        # Minhash equality of element case
        recov_mheq_mean, recov_mheq_std = np.mean(recovered_mh_eq_ckpt_set_len), np.std(recovered_mh_eq_ckpt_set_len)
        print(f'Avg len signature collision case (at least one equality) {recov_mheq_mean} +- {recov_mheq_std}')
        # If filter out 0
        recovered_mh_eq_ckpt_set_len_wo0 = [c for c in recovered_mh_eq_ckpt_set_len if c != 0]
        recov_mheq_mean0, recov_mheq_std0 = np.mean(recovered_mh_eq_ckpt_set_len_wo0), np.std(
            recovered_mh_eq_ckpt_set_len_wo0)
        print(
            f'Avg len signature collision case (at least one equality) when 0 filtered out {recov_mheq_mean0} +- {recov_mheq_std0}')

        split_id += 1

    print(f'Aggregated statistics')
    # Average ckpt set len
    # Target:
    avg_ckpts_len_target_mean, avg_ckpts_len_target_std = np.mean(ckpt_set_avg_len_target_list), np.std(
        ckpt_set_avg_len_target_list)
    print(
        f'Aggregated avg length of target trajectory checkpoint set is: {avg_ckpts_len_target_mean} +- {avg_ckpts_len_target_std}')
    # When accumulated ratios:
    ratiod_avg_ckpts_len_tgt_mean, ratiod_avg_ckpts_len_tgt_std = np.mean(
        ckpt_set_len_ratio_total_ckpt_avg_list_target), np.std(ckpt_set_len_ratio_total_ckpt_avg_list_target)
    print(f'same but previously ratiod {ratiod_avg_ckpts_len_tgt_mean} +- {ratiod_avg_ckpts_len_tgt_std}')

    # Recovered:
    avg_ckpts_len_recov_mean, avg_ckpts_len_recov_std = np.mean(ckpt_set_avg_len_recovered_list), np.std(
        ckpt_set_avg_len_recovered_list)
    print(
        f'Aggregated avg length of target recovered checkpoint set is: {avg_ckpts_len_recov_mean} +- {avg_ckpts_len_recov_std}')
    ratiod_avg_ckpts_len_rcv_mean, ratiod_avg_ckpts_len_rcv_std = np.mean(
        ckpt_set_len_ratio_total_ckpt_avg_list_recovered), np.std(ckpt_set_len_ratio_total_ckpt_avg_list_recovered)
    print(f'same but previously ratiod {ratiod_avg_ckpts_len_rcv_mean} +- {ratiod_avg_ckpts_len_rcv_std}')
    # When accumulated ratio

    sys.stdout.close()  # close the file

    if LATEX_SAVE:
        # Reset matplotlib params to default:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

