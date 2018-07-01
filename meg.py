"""

Creating functions for basic MNE processing

"""
import numpy as np


def epochs_from_raw(raw, trigs, tlims, ids, mwrep=False, **kwargs):
    """

    This function reads in raw data and outputs an Epochs class based on
    triggers, time limits and offsets from the trigger to be placed at 0

    raw: instance of class Raw from mne
    trigs: list of trigger values to include in events
    offset: either a scalar, a set amount to offset the trigger, or array_like,
            the list must be as long as the number of events
    ids: The names for each event type

    """
    from mne import find_events, pick_types, Epochs
    if type(trigs) is not list:
        TypeError("Trigger must be of type list, not {}".format(type(trigs)))

    events = find_events(raw, consecutive=True, shortest_event=1,
                         verbose=False)
    good_evs = np.in1d(events[:, 2], trigs)
    events = events[good_evs]

    if mwrep:
        if len(events) == 141:
            events = np.delete(events, 0, axis=0)

    if len(trigs) is not len(ids):
        raise ValueError(("ids should have same length as trigs. Currently, {}"
                         "and {}").format(len(ids), len(trigs)))
    event_ids = dict(zip(ids, trigs))
    picks = pick_types(raw.info)

    return Epochs(raw, events, event_id=event_ids, tmin=tlims[0],
                  tmax=tlims[1], picks=picks, **kwargs)


def thresh_epochs(epochs, thresh, baseline=(None, None), toi=(None, None)):
    """

    This function takes in epochs and outputs the timing that the signal
    reaches a specific threshold. It's first use is for EMG electrodes to
    detect the onset of muscle activity.

    """
    # check that epochs is an instance of MNE's epochs
    from mne import BaseEpochs
    if not isinstance(epochs, BaseEpochs):
        raise TypeError('First argument must be of mne''s Epochs.'
                        ' Instead it is {}'.format(type(epochs)))

    if not isinstance(thresh, (int, float)):
        raise TypeError('Second argument must be a scalar to determine'
                        'threshold. Instead it is {}'.format(type(thresh)))

    if baseline[0] is None:
        baseline = (np.min(epochs.times), baseline[1])

    if baseline[1] is None:
        baseline = (baseline[0], np.max(epochs.times))

    if toi[0] is None:
        toi = (np.min(epochs.times), toi[1])

    if toi[1] is None:
        toi = (toi[0], np.max(epochs.times))

    trials = epochs.get_data()
    if trials.shape[1] != 1:
        import warnings
        warnings.warn('Data contains more than one channel, computing rms')
        trials = np.sqrt(np.mean(trials**2, axis=1))
    else:
        trials = np.squeeze(trials)

    t_ind = np.logical_and(epochs.times >= baseline[0],
                           epochs.times <= baseline[1])
    mu = np.mean(trials[:, t_ind], axis=1)
    std = np.std(trials[:, t_ind], axis=1)

    ztrials = (trials-mu[:, None])/std[:, None]
    thresh = 6

    time_bool = np.logical_and(epochs.times[None, :] >= toi[0],
                               epochs.times[None, :] <= toi[1])
    zthresh = np.logical_and(ztrials > thresh, time_bool)

    ind = np.argmax(zthresh, axis=1)
    ind *= np.max(zthresh) == 1

    timetomuscle = epochs.times[ind]
    timetomuscle[ind == 0] = np.nan

    return timetomuscle


def trigs_from_raw(raw_data, raw_trigs):
    """

        Designed for yokogawa system.
    """
    from mne.io import Raw, RawArray
    import os.path as op
    from warnings import warn

    if isinstance(raw_data, str):
        if op.exists(raw_data):
            raw_data = Raw(raw_data, preload=True, verbose=False)
        else:
            raise ValueError('Could not find path to raw data, {}'.format(
                raw_data))
    elif not isinstance(raw_data, Raw):
        raise TypeError('raw_data must be either an instance of Raw or path'
                        'to raw file.')

    if isinstance(raw_trigs, str):
        if op.exists(raw_trigs):
            raw_trigs = Raw(raw_trigs, preload=True, verbose=False)
        else:
            raise ValueError('Could not find path to raw data, {}'.format(
                raw_trigs))
    elif not isinstance(raw_trigs, Raw):
        raise TypeError('raw_trigs must be either an instance of Raw or path'
                        'to raw file.')

    # ensure that the sampling rate is the same across both
    if raw_data.info['sfreq'] != raw_trigs.info['sfreq']:
        warn(('Sampling rates were not the same, data is {} but triggers'
             ' are {}. Resampling to the lower option now').format(
             raw_data.info['sfreq'], raw_trigs.info['sfreq']))
        if raw_trigs.info['sfreq'] > raw_data.info['sfreq']:
            raw_trigs.resample(raw_data.info['sfreq'])
        else:
            raw_data.resample(raw_trigs.info['sfreq'])

    # get the cleaned data and the raw_data with extra channels
    cldat = raw_data[:][0]
    rwdat = raw_trigs[:][0]
    # put the cleaned data into the raw data
    rwdat[:157, :] = cldat[:157, :].copy()
    # make new raw array with info from the more informative
    return RawArray(rwdat, raw_trigs.info, first_samp=0, verbose=False)


def evoked_movie(evoked, times=None, interval=200, vtime=None):
    """
    Take evoked array and make into a movieself.
    evoked = evoked data to be animated
    times = the time point of each successive frame
    interval = the time in ms between each frame.
    vtime = time in seconds to base colorbar on
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if times is None:
        times = evoked.times
    # get colorbar limits by time
    if vtime is None:
        t_ind = np.in1d(evoked.times, times)
    else:
        t_ind = np.abs(evoked.times - vtime).argmin()
    # get max abs based on the acceptable times
    vmax = np.abs(evoked.data[:, t_ind]).max()*10**15
    vmin = -vmax

    fig, ax = plt.subplots()

    def animate(t):
        c = evoked.plot_topomap(times=t, time_unit='s', size=2, colorbar=False,
                                axes=ax, contours=0, vmin=vmin, vmax=vmax,
                                show=False)
        return iter(c.axes[0].get_children())

    ani = FuncAnimation(fig, animate, times, interval=interval)
    return ani
