# Providing functions to load ECG data (single or multi lead) from the LUDB and plot the data with annotations.
import wfdb
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

def load_ecg(record_name="1", lead='i', filtered=False):
    """
    Function to load an (example) ECG record from the LUDB.

    Params:
        record_name (str): The name of the ECG record to load.
        lead (str): The lead of the ECG record to load.

    Returns:
        time (ndarray): Array containing the time values of the ECG signal.
        ecg (ndarray): Array containing the preprocessed ECG signal.
        fs (int): The sampling frequency of the ECG signal.
        annotations (dict): Dictionary containing the annotations of the ECG signal.

    """

    def _get_peak_on_off(a, symbols=["N"], left=0):
        ''' Auxiliary function to extract  on/peak/offset of annotations given the following format: ´["(", "N", ")", "(", "N"...]´ '''
        a_sym = np.array(a.symbol)

        # find peak and its on/offset
        peak_ix = np.where(np.in1d(a_sym, symbols))[0]
        on_ix = peak_ix - 1
        off_ix = peak_ix + 1

        # check bounds
        off_ix = off_ix[off_ix < len(a_sym)]
        on_ix = on_ix[on_ix >= 0]

        # check if it is really on/offset == "(" or ")"
        on = a.sample[on_ix[a_sym[on_ix] == "("]] - left
        off = a.sample[off_ix[a_sym[off_ix] == ")"]] - left
        peaks = a.sample[peak_ix] - left

        return on, peaks, off

    # Load the ECG data
    record = wfdb.rdsamp(record_name, pn_dir="ludb/1.0.1/data")
    lead_index = record[1]['sig_name'].index(lead)
    ecg = record[0][:, lead_index]
    fs = record[1]["fs"]
    time = np.arange(ecg.size) / fs

    # Clean the ECG
    if filtered:
        ecg = nk.ecg_clean(ecg, sampling_rate=fs)

    # Load annotations
    annotations = {}
    a = wfdb.rdann(record_name, lead, pn_dir="ludb/1.0.1/data")
    annotations["P_on"], annotations["P"], annotations["P_off"] = _get_peak_on_off(a, symbols=["p"])
    annotations["R_on"], annotations["R"], annotations["R_off"] = _get_peak_on_off(a, symbols=["N"])
    annotations["T_on"], annotations["T"], annotations["T_off"] = _get_peak_on_off(a, symbols=["t"])

    return time, ecg, fs, annotations


def load_multilead_ecg(record_name="1", leads=['i', 'ii', 'iii'], filtered=False):
    """
    Function to load multi-lead ECG data from the LUDB.

    Params:
        record_name (str): The name of the ECG record to load.
        leads (list): List of leads to load.

    Returns:
        time (array-like): Array of time values for the ECG data.
        multilead_ecg (list): List of arrays containing the loaded ECG data for each lead.
        fs (float): Sampling frequency of the ECG data.
        multilead_annotations (dict): Dictionary consisting of a list for each wave, containing each the annotations for each lead.
        
    """
    multilead_annotations = {
        'P': [],
        'R': [],
        'T': [],
        'P_on': [],
        'P_off': [],
        'R_on': [],
        'R_off': [],
        'T_on': [],
        'T_off': []
    }
    multilead_ecg = []

    # Load the ECG data for each lead
    for lead in leads:
        time, ecg, fs, annotations = load_ecg(record_name, lead, filtered=filtered)
        # Append the ECG data
        multilead_ecg.append(ecg)
        # Append the annotations
        for wave, annotation in annotations.items():
            multilead_annotations[wave].append(annotation)

    return time, multilead_ecg, fs, multilead_annotations


# Functions to plot ECG data (single or multi lead) with annotations

def plot_waves(time, ecg, waves, title=None, xlim=[1,8]):
    """
    Plots a single ECG lead with morphology waves.

    Params:
        time (array-like): Array of time indices.
        ecg (array-like): Array of ECG signal.
        waves (dict): Dictionary containing morphology waves.
        title (str, optional): Title of the plot.
    """

    # Plotting options
    markers = [
            {'marker': '^', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 1},
            {'marker': 'x', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 2},
            {'marker': 'v', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 1}]
    colors = ['tab:blue',  'tab:green', 'tab:red' ,'tab:orange', 'tab:brown', 'tab:purple']

    morphologies = ["P", "Q", "R", "S", "T"]
    types = ["_on", "", "_off"]

    # Plot the ECG
    plt.figure(figsize=(10, 4))
    plt.plot(time, ecg, lw=0.75, color=colors[0])
    # iterate over all morphology waves
    for i, w in enumerate(morphologies):
        for t, marker in zip(types, markers):
            if not f"{w}{t}" in waves.keys():
                continue
            plt.plot(time[waves[f"{w}{t}"]], ecg[waves[f"{w}{t}"]], **marker, label=f"{w}{t}", color=colors[i+1])

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend(loc='upper right', title='waves')
    plt.title(title)
    plt.grid(True)
    plt.xlim(xlim)
    plt.show()

def plot_multilead_ecg_annotations(time, multilead_ecg, multilead_annotations, leads, xlim=[1,8]):
    markers = [
            {'marker': '^', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 1},
            {'marker': 'x', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 2},
            {'marker': 'v', 'linewidth': 0, 'markersize': 6, 'markeredgewidth': 1}]
    colors = ['tab:blue',  'tab:green', 'tab:red' ,'tab:orange', 'tab:brown', 'tab:purple']

    morphologies = ["P", "Q", "R", "S", "T"]
    types = ["_on", "", "_off"]

    fig, axs = plt.subplots(len(leads), 1, figsize=(10, 2 * len(leads)), sharex=True)

    for i, lead in enumerate(leads):
        axs[i].plot(time, multilead_ecg[i], lw=0.75, color=colors[0])
        axs[i].set_ylabel('Amplitude')
        axs[i].set_title(f'Lead {lead}')

        # Plot all morphology waves
        for k, w in enumerate(morphologies):
            for t, marker in zip(types, markers):
                if not f"{w}{t}" in multilead_annotations.keys():
                    continue
                label = label=f"{w}{t}" if i == 0 else None
                axs[i].plot(time[multilead_annotations[f"{w}{t}"][i]], multilead_ecg[i][multilead_annotations[f"{w}{t}"][i]], **marker, color=colors[k+1], label=label)

        
        axs[i].set_xlim(xlim)
        axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, -1.0), title='waves')

    plt.show()