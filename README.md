![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue.svg)

# OTAnalysis

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
```{}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n newenv
conda activate newenv

```
## Dependencies
```{}
conda install numpy
conda install pandas
conda install scipy
pip install PyQt5
conda install pyqt qtpy
conda install matplotlib
```

## Launch project
```{}
cd packages
python main.py
```

## Documentation

### Update
```{}
make html
```

### Visualization
```{}
firefox docs/_build/html/index.html
```

### Explanation output file
output file in the form of a table of 48 columns on the number of curves of the analysis (1 curve per line)
columns
#### Important data from the analysis for post-processing
- treat_supervised:
- automatic_type
- type
- automatic_AL
- AL
- automatic_AL_axe

#### Data of the analysis parameters
- model
- Date
- Hour
- condition
- drug
- tolerance
- bead
- cell

#### Theoretical data present in the headers of the files
- main_axis
- stiffness
- theorical_contact_force (N)
- theorical_distance_Press (m)
- theorical_speed_Press (m/s)
- theorical_freq_Press (Hz)
- time_segment_pause_Wait1 (s)
- theorical_distance_Pull (m)
- theorical_speed_Pull (m/s)
- theorical_freq_Pull (Hz)

#### Data calculated during the analysis
- baseline_press (pN)
- std_press (pN)
- slope (pN/nm)
- error (pN/nm)
- contact_point_index
- contact_point_value  (pN)
- force_min_press_index
- force_min_press_value (pN)
- point_release_index
- point_release_value (pN)
- force_max_pull_index
- force_max_pull_value (pN)
- point_return_endline_index
- point_return_endline_value
- Pente (pN/nm)

#### Data calculated if curves different from non-adhesive
- point_transition_index
- point_transition_value (pN)
- jump_force_start_pull (pN)
- jump_distance_start_pull (nm)
- jump_distance_end_pull (nm)
- jump_force_end_pull (pN)

#### Boolean validation of the fits 
- valid_fit_press
- valid_fit_pull










