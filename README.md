![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue.svg)

## Tool for managing the results of optical tweezers
Tool for extracting, analyzing and classifying optical tweezer data curves

### Flow of the use process
Launch on an interface allowing to select the parameters for a file analysis in back-end

#### Condition of the experience
- condition: Name of the antibody present on the beads during the experiment
- drug: name of the drug used for the analysis if present

#### Fitting management
Selection of the curve files to be analyzed either by selecting a directory or with the multiple selection of files
- model: The model used to fit the "Press" curves (linear or sphere)
If selected sphere, appearance of the physical parameters menu for the calculation of the Young's modulus
- eta: Poisson's ratio
- bead rdius: diameter of the ball used during the experiment

#### Management of curve anomalies 
Curve management parameters Incomplete (no analysis possible) or misaligned (analysis but warning)
- pulling length min : minimum percentage of the length of the "Pull" segment to determine if the curve is rejected despite the presence of all the segments indicated in the header
- Fmax epsilon: percentage of max force on the major axis to determine misalignment of the curve on the minor axes

#### Classification condition
- NAD if jump is < (PN): force condition to classify non-adhesive curves < pN
- AD if position is < (nm): distance condition to separate the membership from the tubes
- AD if slope is < (pts): condition number of points to separate the membership of the tubes
- Factor overcome noise (xSTD): Number of times the standard deviation for the calculation of the characteristic points
- Factor optical effect (xSTD): Number of times the standard deviation to correct the optical effect

Appearance of a method loading button after loading data to redo a past analysis.

### Menu after launching the analysis
Three possible options:
- Supervised: Allows you to switch to a new window with the display of curves and a supervisory menu
- Unsupervised: Allows you to retrieve the output file of the automatic analysis
- ...with graphs: Allows you to retrieve the output file of the automatic analysis completed with all the graphs

If we choose supervised:
### Graphic display window with supervision
Visualization of all curves as a function of time on the 3 axes and as a function of distance on the main_axis

#### Supervision menu
- Close supervision Panel: Possibility to close this menu for a more important visualization of the curves
- Buttons to navigate between the curves. Can be operated with the left/right arrow keys
- Button to save the displayed graphic
- Button to save the output file with an indication that the supervision is stopped at this curve (treat_supervised column)
- curve name
- button for zooming with characteristic point and fit on distance curves (Pull and Press segment only)
- if curve misaligned warning of misalignment axis with a possibility to change this status
- fit validation of the Press segment
- management of the optical correction with a post-analysis control
- fit validation of the Pull segment
- correction of the type of curve classification (type defined checked)
- Pagination to determine our position in the whole analysis. Possibility to move with the 'Enter' key and the number of the curve

## Install MiniConda && Create conda environment
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n newenv
conda activate newenv

```
## Dependencies
```
conda install numpy
conda install pandas
conda install scipy
pip install PyQt5
conda install pyqt qtpy
conda install matplotlib
pip install supyr-struct
pip install pathlib
pip install pytest

```

## Install package
You can install napari-openlabframe via <a href="https://pypi.org/project/pip/" rel="noopener" target="_blank">pip</a>:
```
python -m pip install -e
```

## Launch project
```
python -m src.main
```

## Documentation

### Update
```
make html
```

### Visualization
```
firefox docs/_build/html/index.html
```

### Explanation output file
output file in the form of a table of 48 columns on the number of curves of the analysis (1 curve per line)
columns
#### Important data from the analysis for post-processing
- treat_supervised: bool
    True if curve visualized otherwise False
- automatic_type: str
    type determined by the automatic analysis
- type: str
    type given to the curve with the supervisor menu. If there is no supervision then the same as the 'automatic_type' column.
- automatic_AL: str
    "No" if the curve is misaligned according to the automatic threshold otherwise "Yes"
- AL: str
    Readjustment through supervision. If no supervision, then same as "automatic_AL"
- automatic_AL_axe: list
    secondary axis affected by curve misalignment and its sign to know the direction of the misalignment with respect to the direction of the main axis
- optical_state: str
    optical correction applied (No_correction, Auto_correction, Manual_correction)


#### Data of the analysis parameters
- model: str
    model for the fit on "Press" segment chosen by the user for the analysis
- Date: str
    date of creation of the curve file
- Hour: str
    time of creation of the curve file
- condition: str
    condition applied to the analysis set (often antibodies on the bead)
- drug: str
    drug put in the medium for analysis (can be used to add a second condition)
- tolerance: float
    noise tolerance for the baseline (xstd)
- bead: str
    number of the ball used for the curve
- cell: str
    number of the cell used for the curve
- couple: str
    couple ball number and cell number

#### Theoretical data present in the headers of the files
- main_axis: str
    main axis of the experiment and the direction of approach of the cell with respect to the ball:
        +X: the cell approaches from the right 
        -X : the cell approaches from the left
        +Y : the cell comes from the top
        -Y : the cell comes from the bottom
- stiffness: float
    value of the spring stiffness to correct the distance values
- theorical_contact_force (N): float
    theoretical contact force between the ball and the cell required by the user before starting the experiment 
- theorical_distance_Press (m): float
    theoretical length of the "Press" segment
- theorical_speed_Press (m/s): float
    theoretical speed of the "Press" segment 
- theorical_freq_Press (Hz): float
    theoretical frequency of the "Press" segment 
- time_segment_pause_Wait1 (s): float
    pause time of the "Wait" segment (often 0s)
- theorical_distance_Pull (m): float
    theoretical length of the "Pull" segment
- theorical_speed_Pull (m/s): float
    theoretical speed of the "Pull" segment 
- theorical_freq_Pull (Hz): float
    theoretical frequency of the "Pull" segment


#### Data calculated during the analysis
- baseline_origin_press (N): float
    average of the first 1000 points of the "Press" segment on the data without correction
- baseline_corrected_press (pN): float
    average of the first 1000 points of the "Press" segment on the data corrected to bring the baseline centered on 0
- std_origin_press (N): float
    standard deviation of the first 1000 points to define the noise rate of the curve (on the data without correction)
- std_corrected_press (pN): float
    standard deviation of the first 1000 points to define the noise rate of the curve (on the data correction)
- slope (pN/nm): float
    calculation of the force slope for the "Press" segment
- error (pN/nm): float
    calculates the error of the force slope for the "Press" segment
- contact_point_index: int
    index of the contact point between the ball and the cell on the "Press" segment
- contact_point_value  (pN): float
    force value of the contact point between the ball and the cell on the "Press" segment
- force_min_press_index: int
    index of the minimum force of the "Press" segment
- force_min_press_value (pN): float
    value of the minimum force of the "Press" segment
- force_min_curve_index: int
    index of the minimum force of the curve (sometimes confused with minimum Press)  
- force_min_curve_value (pN): float
    value of the minimum force of the curve (sometimes confused with minimum Press)
- point_release_index: int
    'index of the point where the cell loses contact with the ball (without taking \ into account the adhesive molecules or the membrane tubes).'
- point_release_value (pN): float 
    value of the point where the cell loses contact with the ball (without taking \ into account the adhesive molecules or the membrane tubes).
- force_max_pull_index: int
    index of the maximum force on a part of the "Pull" segment between the release point and the return to the baseline
- force_max_pull_value (pN): float
    value of the maximum force on a part of the "Pull" segment between the release point and the return to the baseline
- force_max_curve_index: int
    index of the maximum force of the curve
- force_max_curve_value (pN) : float
    value of the maximum force of the curve
- Pente (pN/nm): float
    coefficient of the contact loss slope between the ball and the cell due to the retraction effect of the cell with respect to the ball

#### Data calculated if type of curves different from non-adhesive, infinite tube or rejected
- point_transition_index: int
    index of the break point of the tube (called transition point)
- point_transition_value (pN): float
    value of the break point of the tube (called transition point)
- point_return_endline_index: int
    index of the point where the curve returns to the baseline values
- point_return_endline_value: float
    value of the point where the curve returns to the baseline values

**Jumps:**
- jump_force_start_pull (pN): float
    force jump between the release point and the maximum force of the curve in the case of an adhesion or a finished tube
- jump_force_end_pull (pN): float
    force jump between the maximum force of the curve and the point of return to the baseline
- jump_nb_points
    number of points between the point of return to the baseline and the maximum strength of the curve 
- jump_time_start_pull (s)
    time between the release point and the maximum force of the curve
- jump_time_end_pull (s)
    time between the maximum force of the curve and the point of return to the baseline
- jump_distance_start_pull (nm): float
    distance between the release point and the maximum force of the curve
- jump_distance_end_pull (nm): float
    distance between the maximum force of the curve and the point of return to the baseline

#### Boolean validation of the fits 
- valid_fit_press : bool
    validation of the fit on the "Press" segment. False by default because not validated

- valid_fit_pull : bool
    validation of the fit on the "Pull" segment. False by default because not validated










