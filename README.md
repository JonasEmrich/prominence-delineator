# Physiology-Informed ECG Delineation Based on Peak Prominence

This Python package implements the Peak Prominence ECG Delineator \[1\] and provides methods for ECG cleaning and R-peak detection \[2\], resulting in a complete delineation pipeline. This delineator allows for fast and precise detection of the positions, on- and offsets of morphology waves (e.g., _P_, _R_, _T_) in single or multi-lead ECG signals. An optional multi-lead correction procedure can be applied, leveraging information from all leads if available. 

## Advantages and Limits
This proposed approach achieves a highly explainable and interpretable wave selection by leveraging prominence information. Hence, wave detection only depends on physiologically motivated parameters chosen so that morphologies of interest are well represented and portrayed, yielding high $F_1$-scores and low errors on established Datasets in comparison to competing methods \[1\]. 

**NOTE:** This approach allows for further customization w.r.t to these parameters so that different parameter choices or other physiologically informed decision rules might result in higher performance or robustness regarding certain morphologies and heartbeat types. Even though, the utilized approach for on- and offset detection yielded great performance it is constrained by typical physiological boundaries. Developing novel prominence computation methods to robustly identify basepoints might therefore yield further improvements. 



## Installation

You can install the latest version of the `prominence-delineator` package from the Python Package Index (PyPI) by running:
```
pip install prominence-delineator
```

## Usage

A complete working example is provided in [example.ipynb](https://github.com/JonasEmrich/prominence-delineator/blob/main/example/example.ipynb), additionally the basic usage is depicted below.
The ProminenceDelineator takes the `sampling_frequency` and optionally several physiological parameters as input.
Then, R-peaks can be detected using any reliable R-peak detector or with the integrated method (applying the `FastNVG` \[2\] approach). Before detecting further waves, the required ECG cleaning can easily be performed by using `.clean_ecg(ecg)`.
Finally, morphology waves can be detected using `.find_waves()` or `.find_waves_multilead()` for single or multi-lead ECG signals, respectively. 

**Note:** For multi-lead delineation, the `ecg` and `rpeaks` input should be in the form of a list or ndarray containing the ECG signals or R-peaks of each lead, respectively. The output will be given as a dictionary with keys denoting the wave type and values lists containing wave positions or lists of wave positions for all leads when processing multi-lead data.

### Single Lead ECG Delineation
``` python
from prominence_delineator import ProminenceDelineator 

# Create an instance of the ProminenceDelineator
PromDelineator = ProminenceDelineator(sampling_frequency=fs)
# Detect the R-peaks in the ECG signal
rpeaks = PromDelineator.find_rpeaks(ecg)
# Clean the ECG signal
ecg = PromDelineator.clean_ecg(ecg)
# Find waves in the ECG signal using the ProminenceDelineator
waves = PromDelineator.find_waves(ecg, rpeaks=rpeaks)
```

![Delineated ECG](https://github.com/JonasEmrich/prominence-delineator/blob/main/img/plot.png)

### Multi-Lead ECG Delineation
``` python
from prominence_delineator import ProminenceDelineator 

# Create an instance of the ProminenceDelineator
PromDelineator = ProminenceDelineator(sampling_frequency=fs)
# Detect the R-peaks in the multilead ECG signal
multilead_rpeaks = PromDelineator.find_rpeaks(ecg_multilead)
# Clean the ECG signal
ecg_multilead = PromDelineator.clean_ecg(ecg_multilead)
# Find waves in the ECG signal using the find_waves function
multilead_waves = PromDelineator.find_waves_multilead(ecg_multilead, rpeaks_multilead=multilead_rpeaks)
```

## References
* \[1\] Emrich, J., Gargano, A., Koka, T., and Muma, M. (2024), Physiology-Informed ECG Delineation Based on Peak Prominence.
* \[2\] Emrich, J., Koka, T., Wirth, S. and Muma, M. (2023). Accelerated Sample-Accurate R-Peak Detectors Based on Visibility Graphs.

