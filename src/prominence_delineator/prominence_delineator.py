import numpy as np
import scipy


class ProminenceDelineator:
    """A class implementing a prominence based delineation algorithm [1], detecting morphology waves (P, Q, R, S, T) in
    an ECG signal. This class allows for processing single and multi-lead ECG signals and incorporates functionalities
    for ECG cleaning, R-peak detection [2] and muli-lead wave correction, providing a complete ECG delineation pipeline.

    Args
    -----
    sampling_frequency : float
        The sampling frequency of the ECG signal.
    max_r_rise_time : float, optional
        The maximum rise time for the R wave (in seconds). Default is 0.12.
    typical_st_segment : float, optional
        The typical duration of the ST segment (in seconds). Default is 0.15.
    max_pr_interval : float, optional
        The maximum duration of the PR interval (in seconds). Default is 0.3.
    max_qrs_interval : float, optional
        The maximum duration of the QRS interval (in seconds). Default is 0.18.

    References
    -----------
    [1] Emrich, J., Gargano, A., Koka, T., and Muma, M. (2024), "Physiology-Informed ECG Delineation
    Based on Peak Prominence", 32nd European Signal Processing Conference (EUSIPCO), 2024
    [2] J. Emrich, T. Koka, S. Wirth, M. Muma, "Accelerated Sample-Accurate R-peak Detectors Based on
    Visibility Graphs", 31st European Signal Processing Conference (EUSIPCO), 2023

    """

    def __init__(
        self,
        sampling_frequency,
        max_r_rise_time=0.12,
        typical_st_segment=0.15,
        max_pr_interval=0.3,
        max_qrs_interval=0.18,
    ):

        if sampling_frequency <= 0:
            raise ValueError(
                f"'sampling_frequency' has to be a positive non-zero value (got {sampling_frequency})."
            )
        if max_r_rise_time <= 0:
            raise ValueError(
                f"'max_r_rise_time' has to be a positive non-zero value (got {max_r_rise_time})."
            )
        if typical_st_segment <= 0:
            raise ValueError(
                f"'typical_st_segment' has to be a positive non-zero value (got {typical_st_segment})."
            )
        if max_pr_interval <= 0:
            raise ValueError(
                f"'max_pr_interval' has to be a positive non-zero value (got {max_pr_interval})."
            )
        if max_qrs_interval <= 0:
            raise ValueError(
                f"'max_qrs_interval' has to be a positive non-zero value (got {max_qrs_interval})."
            )
        if sampling_frequency <= 0:
            raise ValueError(
                f"'sampling_frequency' has to be a positive non-zero value (got {sampling_frequency})."
            )

        # physiology parameters motivated by literature
        self.fs = sampling_frequency
        self.max_r_rise_time = int(max_r_rise_time * self.fs)
        self.typical_st_segment = int(typical_st_segment * self.fs)
        self.max_pr_interval = int(max_pr_interval * self.fs)
        self.max_qrs_interval = int(max_qrs_interval * self.fs)

        # basepoint_intervals chosen as narrow intervals since classic prominence calculation is non-robust
        self.max_p_basepoint_interval = 0.1 * self.fs
        self.max_r_basepoint_interval = 0.1 * self.fs
        self.max_t_basepoint_interval = 0.2 * self.fs

    def find_waves_multilead(
        self,
        sig_multilead,
        rpeaks_multilead,
        multilead_correction=False,
        tol=0.15,
        accept_forall_factor=2 / 3,
        accept_factor=1 / 3,
        include_nodetections=False
    ):
        """Delineate ECG waves in multiple leads of a provided ECG signal.

        Args
        ----
        sig_multilead : list
            A list of multiple arrays or lists representing the filtered ECG signal in each lead.
        rpeaks_multilead: list
            A list of multiple arrays or lists representing the R-peaks in each lead.
        multilead_correction : bool, optional
            If True, the detected waves will be corrected based on the detection from the other leads. Defaults to
            False.
        tol : float, optional
            Tolerance factor for correcting wave positions (in seconds). Defaults to 0.15.
        accept_forall_factor : float, optional
            The minimum proportion of leads in which a wave needs to be detected, so that its corrected position is
            introduced in the remaining leads. Only has an effect if 'multilead_correction' is True. Defaults to 2/3.
        accept_factor : float, optional
            The minimum proportion of leads in which a wave needs to be detected so that it will not be discarded. Only
            has an effect if 'multilead_correction' is True. Defaults to 1/3.
        include_nodetections : bool, optional
            If True, the output will include `None` when no wave is found in the current beat/complex.
            This results in equally sized arrays for each wave. Defaults to False.

        Returns
        -------
        dict: A dictionary containing the detected waves in each lead. The keys represent the wave types ('P', 'Q',
        'R', 'S', 'T', 'P_on', 'P_off', 'R_on', 'R_off', 'T_on', 'T_off') and each value is a lists containing
        predicted wave positions (as a list) for each lead.

        """

        if not isinstance(sig_multilead, list) or not all(
            isinstance(lead, (list, np.ndarray)) for lead in sig_multilead
        ):
            raise ValueError(
                "Invalid format for 'sig_multilead'. It should be a list of multiple arrays or lists."
            )

        if len(sig_multilead) != len(rpeaks_multilead):
            raise ValueError(
                "The length of 'sig_multilead' and 'rpeaks_multilead' should be the same, i.e., the number of"
                "provided leads should match."
            )
        if not isinstance(rpeaks_multilead, list) or not all(
            isinstance(lead, (list, np.ndarray)) for lead in rpeaks_multilead
        ):
            raise ValueError(
                "Invalid format for 'rpeaks_multilead'. It should be a list of multiple arrays or lists."
            )
        waves = {
            "P": [],
            "Q": [],
            "R": rpeaks_multilead,
            "S": [],
            "T": [],
            "P_on": [],
            "P_off": [],
            "R_on": [],
            "R_off": [],
            "T_on": [],
            "T_off": [],
        }

        # process all leads
        for l, lead in enumerate(sig_multilead):
            result = self.find_waves(lead, rpeaks=waves["R"][l], include_nodetections=include_nodetections)
            # append result for lead
            for wave in waves.keys():
                if wave in result and wave != "R": # R peaks are already appended
                    waves[wave].append(result[wave])

        # correct multi-lead predictions
        if multilead_correction:
            for wave, leads in waves.items():
                waves[wave] = self._correct_multiLeads(
                    leads,
                    tol=tol,
                    accept_forall_factor=accept_forall_factor,
                    accept_factor=accept_factor,
                )

        return waves

    def find_waves(self, sig, rpeaks, include_nodetections=False):
        """Delineate ECG waves for the given single lead ECG signal.

        Args
        ----
        sig : ndarray
            The input signal representing a single unfiltered ECG lead.
        rpeaks : ndarray
            The R-peak positions of the signal.
        include_nodetections : bool, optional
            If True, the output will include `None` when no wave is found in the current beat/complex.
            This results in equally sized arrays for each wave. Defaults to False.

        Returns
        -------
        dict: A dictionary of ndarrays containing the indices of the detected waves. The keys are 'P', 'R', 'T',
        'P_on', 'P_off', 'R_on', 'R_off', 'T_on', 'T_off'.

        """

        if sig is None:
            raise ValueError("The input signal 'sig' is None.")

        waves = {
            "P": [],
            "Q": [],
            "R": [],
            "S": [],
            "T": [],
            "P_on": [],
            "P_off": [],
            "R_on": [],
            "R_off": [],
            "T_on": [],
            "T_off": [],
        }

        # calculate RR intervals
        rr = self._calc_rr_intervals(rpeaks)

        # initialize variables
        l = 0
        for i in range(len(rpeaks)):
            # 1. split signal into segments
            rpeak_pos = min(rpeaks[i], rr[i] // 2)
            l = rpeaks[i] - rpeak_pos
            r = rpeaks[i] + rr[i + 1] // 2
            s = sig[l:r]

            current_wave = {
                "R": rpeak_pos,
            }

            # 2. find local extrema in signal
            local_maxima = scipy.signal.find_peaks(s)[0]
            local_minima = scipy.signal.find_peaks(-s)[0]
            local_extrema = np.concatenate((local_maxima, local_minima))

            # 3. compute prominence weight
            weight_maxima = self._calc_prominence(local_maxima, s, current_wave["R"])
            weight_minima = self._calc_prominence(
                local_minima, s, current_wave["R"], minima=True
            )

            if local_extrema.any():
                # find waves
                self._find_q_wave(s, weight_minima, current_wave)
                self._find_s_wave(s, weight_minima, current_wave)
                self._find_p_wave(local_maxima, weight_maxima, current_wave)
                self._find_t_wave(
                    local_extrema, (weight_minima + weight_maxima), current_wave
                )
                self._find_on_offsets(
                    s, local_minima, local_maxima, weight_maxima, current_wave
                )

            # append waves
            for key in waves:
                if key == "R":
                    waves[key].append(int(rpeaks[i]))
                elif key in current_wave:
                    waves[key].append(int(current_wave[key] + l))
                elif include_nodetections:
                    waves[key].append(None)

        # cast into np.array int
        if not include_nodetections:
            for key in waves:
                waves[key] = np.array(waves[key], dtype=int)

        return waves

    def clean_ecg(self, sig):
        """Cleaning the given ECG signal.

        Args
        ----
        sig : ndarray
            The input signal representing a single unfiltered ECG lead.

        Returns
        -------
        ndarray: The cleaned ECG signal.

        Note
        ----
        The 'neurokit2' module is required for this method to run. Please install it first (`pip install neurokit2`).

        """
        try:

            import neurokit2 as nk

            if isinstance(sig, (list, np.ndarray)) and all(
                isinstance(lead, (list, np.ndarray)) for lead in sig
            ):
                # multi-lead signal
                return [nk.ecg_clean(lead, sampling_rate=self.fs) for lead in sig]
            else:
                # single lead signal
                return nk.ecg_clean(sig, sampling_rate=self.fs)

        except ImportError:
            raise ImportError(
                "The 'neurokit2' module is required for this method to run.",
                "Please install it first (`pip install neurokit2`).",
            )

    def find_rpeaks(self, sig):
        """Detecting R-peaks using the FastNVG detector.

        Args
        ----
        sig : ndarray
            The input signal representing a single unfiltered ECG lead.

        Returns
        -------
        ndarray: The R-peak positions of the signal.

        Note
        ----
        The 'vg-beat-detectors' module is required for this method to run. Please install it first (`pip install
        vg-beat-detectors`).

        """
        try:
            from vg_beat_detectors import FastNVG

            detector = FastNVG(sampling_frequency=self.fs)

            if isinstance(sig, (list, np.ndarray)) and all(
                isinstance(lead, (list, np.ndarray)) for lead in sig
            ):
                # multi-lead signal
                return [detector.find_peaks(lead) for lead in sig]
            else:
                # single lead signal
                return detector.find_peaks(sig)

        except ImportError:
            raise ImportError(
                "The 'vg-beat-detectors' module is required for this method to run.",
                "Please install it first (`pip install vg-beat-detectors`).",
            )

    def _calc_rr_intervals(self, rpeaks):
        """Calculating RR intervals from R-peaks.
        Intervals at segment boundaries are inferred by pre/succeeding segments.

        """
        rr = np.diff(rpeaks)
        rr = np.insert(rr, 0, min(rr[0], 2 * rpeaks[0]))
        rr = np.insert(rr, -1, min(rr[-1], 2 * rpeaks[-1]))
        return rr

    def _correct_multiLeads(
        self, predictions, tol, accept_forall_factor=2 / 3, accept_factor=1 / 3
    ):
        """Corrects the multi-leads predictions based on a tolerance value."""

        if not (0 <= accept_forall_factor <= 1):
            raise ValueError("The acceptance factor must be between 0 and 1, as it represents the percentage of leads in which a detection must occur to be introduced into all leads.")

        if not (0 <= accept_factor <= 1):
            raise ValueError("The acceptance factor must be between 0 and 1, as it represents the percentage of leads in which a detection must occur to be retained in the lead where it was found.")

        predictions = [np.array(lead).flatten() for lead in predictions]

        corr_predictions = [[] for lead in predictions]
        for m in range(len(predictions)):
            for p in predictions[m]:
                Found = [p]
                Found_in = [m]
                for l in range(m + 1, len(predictions)):
                    diff = np.abs(predictions[l] - p).astype(float)
                    i = diff <= tol * self.fs
                    if i.any():
                        Found.extend(predictions[l][i])
                        Found_in.append(l)
                        predictions[l] = predictions[l][~i]

                # add corrected annotation to all leads if found in enough leads
                if len(Found) >= len(predictions) * accept_forall_factor:
                    for c in range(len(predictions)):
                        corr_predictions[c].append(int(np.mean(Found)))
                # add corrected annotation only to the lead where it was found
                elif len(Found) >= len(predictions) * accept_factor:
                    for cx, c in enumerate(Found_in):
                        corr_predictions[c].append(int(Found[cx]))

        return [np.sort(lead) for lead in corr_predictions]

    def _correct_peak(self, sig, peak, window=0.02):
        """Correct peak towards local maxima within provided window."""

        l = peak - int(window * self.fs)
        r = peak + int(window * self.fs)
        if len(sig[l:r]) > 0:
            return np.argmax(sig[l:r]) + l
        else:
            return peak

    def _calc_prominence(self, peaks, sig, Rpeak=None, minima=False):
        """Returns an array of the same length as sig with prominences computed for the provided peaks.
        Prominence of the R-peak is excluded if the R-peak position is given.
        """
        w = np.zeros_like(sig)

        if len(peaks) < 1:
            return w
        # get prominence
        _sig = -sig if minima else sig
        w[peaks] = scipy.signal.peak_prominences(_sig, peaks)[0]
        # optional: set rpeak prominence to zero to emphasize other peaks
        if Rpeak is not None:
            w[Rpeak] = 0
        return w

    def _find_q_wave(self, s, weight_minima, current_wave):
        if not "R" in current_wave:
            return
        q_bound = max(current_wave["R"] - self.max_r_rise_time, 0)

        current_wave["Q"] = (
            np.argmax(weight_minima[q_bound : current_wave["R"]]) + q_bound
        )

    def _find_s_wave(self, s, weight_minima, current_wave):
        if not "Q" in current_wave:
            return
        s_bound = current_wave["Q"] + self.max_qrs_interval
        s_wave = (
            np.argmax(weight_minima[current_wave["R"] : s_bound] > 0)
            + current_wave["R"]
        )
        current_wave["S"] = (
            np.argmin(s[current_wave["R"] : s_bound]) + current_wave["R"]
            if s_wave == current_wave["R"]
            else s_wave
        )

    def _find_p_wave(self, local_maxima, weight_maxima, current_wave):
        if not "Q" in current_wave:
            return
        p_candidates = local_maxima[
            (current_wave["Q"] - self.max_pr_interval <= local_maxima)
            & (local_maxima <= current_wave["Q"])
        ]
        if p_candidates.any():
            current_wave["P"] = p_candidates[np.argmax(weight_maxima[p_candidates])]

    def _find_t_wave(self, local_extrema, weight_extrema, current_wave):
        if not "S" in current_wave:
            return
        t_candidates = local_extrema[
            (current_wave["S"] + self.typical_st_segment <= local_extrema)
        ]
        if t_candidates.any():
            current_wave["T"] = t_candidates[np.argmax(weight_extrema[t_candidates])]

    def _find_on_offsets(
        self, s, local_minima, local_maxima, weight_maxima, current_wave
    ):
        if "P" in current_wave:
            _, p_on, p_off = scipy.signal.peak_prominences(
                s, [current_wave["P"]], wlen=self.max_p_basepoint_interval
            )
            if not np.isnan(p_on):
                current_wave["P_on"] = p_on[0]
            if not np.isnan(p_off):
                current_wave["P_off"] = p_off[0]

        if "T" in current_wave:
            p = -1 if np.isin(current_wave["T"], local_minima) else 1

            _, t_on, t_off = scipy.signal.peak_prominences(
                p * s, [current_wave["T"]], wlen=self.max_t_basepoint_interval
            )
            if not np.isnan(t_on):
                current_wave["T_on"] = t_on[0]
            if not np.isnan(t_off):
                current_wave["T_off"] = t_off[0]

        # correct R-peak position towards local maxima (otherwise prominence will be falsely computed)
        r_pos = self._correct_peak(s, current_wave["R"])
        _, r_on, r_off = scipy.signal.peak_prominences(
            s, [r_pos], wlen=self.max_r_basepoint_interval
        )
        if not np.isnan(r_on):
            current_wave["R_on"] = r_on[0]

        if not np.isnan(r_off):
            current_wave["R_off"] = r_off[0]
