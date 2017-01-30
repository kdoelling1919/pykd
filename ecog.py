"""

functions for ECoG data

"""
import numpy as np
from mne import create_info
from mne.channels import DigMontage


def info_from_montage(filename, raw_chs=None, remove_depth=False,
                      change_ch_names=True, return_montage=False):
    montage = list()
    with open(filename, 'r') as f:
        for line in f:
            montage.append(line.split())
    montage = np.array(montage)

    rem = np.array([False]*len(montage))
    if change_ch_names:
        # convert to format that fits with mne channel info
        begin = ['EEG ', '_']
        ending = '-REF'
        ch_names = np.array([''.join([begin[0], x[:-2], begin[1]])+x[-2:] +
                             ending for x in montage[:, 0]])

    if raw_chs is not None:
        # only use channel names that are in the raw data
        out = np.setdiff1d(ch_names, raw_chs)
        rem = np.in1d(ch_names, out)

    if remove_depth:
        # remove depth electrodes. Using only grids and strips
        depths = np.array([x.find('DAT') + x.find('DPT') > 0
                           for x in ch_names])
        rem = np.logical_or(rem, depths)

    ch_names = ch_names[np.logical_not(rem)]
    montage = montage[np.logical_not(rem)]

    elec = montage[:, 1:4].astype(np.float)/1000
    dig_ch_pos = dict(zip(ch_names, elec))
    mon = DigMontage(dig_ch_pos=dig_ch_pos)

    info = create_info(list(ch_names), 512, 'ecog', montage=mon)
    if return_montage:
        return info, mon
    else:
        return info
