# sharpy_track
 A napari plugin (planned) for mouse brain slice image registration.

## How to install
1. Download this repository by

    >git clone https://github.com/hejDMC/sharpy_track

    in your terminal, make sure you have access to it.

2. Create a conda environment named sharpytrack_env, or whatever you want

    >conda create -n sharpytrack_env python  

    install required packages
    
    >conda install pyqt,opencv,scikit-image,natsort,pandas,matplotlib

## Before first run
>Make sure you have these 2 files under **`./sharpy_track/atlas`**
>- **`template_volume_8bit.npy`**  
>- **`annotation_volume_10um_by_index.npy`**  

It's also nice to have `template_volume_10um.npy` in the same folder, but not necessary. 

## To run the package
In terminal, go to repository containing directory, activate conda environment

>conda activate sharpytrack_env

run sharpy_track

>python sharpy_track

# Brain slice registration workflow
1. Launch sharpy_track
2. Click: **`File > Open Folder`** or **`Ctrl+O`** to select your sample image folder.

>Note: sample image needs to fulfil the following requirements:
>- Dimension: 1140 px width, 800 px height, 8-bit grayscale
>- Format: *.tif, or *.tiff
>- Name: reasonably ordered (to be sorted by `natsort`)

3. Load previous registration record, if there is any.  
Sharpy_track will try to find `registration.json` file inside the path user provided, and remind user to load the record.  
In most cases, user chooses **`Yes`** to load previous registration record.

>Note: if the user chooses **`No`**, before any change has been made, it's still possible to load previous record, by clicking **`File > Load JSON`** or **`Ctrl+L`**.  

>**IMPORTANT**  
Previous record will be **`OVERWRITTEN AND LOST`**, if any **new transformation happens** after user has chosen **`No`** and have not loaded the record manually. 

4. Find a good cut in atlas volume, correspondent to sample image 
>More detail in `Usage` section
5. Click **`transformation toggle`** or press **`T`** to turn on transformation mode.
>`transformation toggle` is a button in the middle, when turned **`ON`** appears `green`, **`OFF`** appears `red`. 

>**IMPORTANT**  
Be careful when you turn on transformation mode when current sample is already registered.  
Turn on transformation mode will **`OVERWRITE`** current `AP location`, `ML angle`, `DV angle` to current sample.

6. When transformation mode is `ON`, **`left click`** on **left viewer** to add paired dots.
>**`Right click`** on **left viewer** removes the most recently added dots pair 

7. Click and **drag** to move the dot to desired location
>Hover over a dot will change it's color to red, while also changing its paired dot color to red.

8. Repeat step **5.** and **6.** until there are at least **`5 pairs of dots`** to see the transformation.

9. **Move** dots or **add** more dots to align the regristration. Press **`Z`**, **`X`**, **`C`** to switch view between **`100% Atlas`**, **`50% Blend`**, **`100% Sample`** modes. Press **`A`** to show or hide brain region outline. 

>Registration progress is **`automatically saved`** whenever there is **a change in any transformation**. So that the user do not need to manually save along the way.  
Registration record is saved to a file named `registration.json`, in the same folder where sharpy_track load sample images.

10. Looks good, **`turn off transformation mode`** and move to the next sample. Don't know what to do next? Please go back to step **4.**

11. Registration finished, **`close`** sharpy_track by clicking `X` at top right window, or **`Ctrl+Q`**, or **`File > Exit`**


# Usage
## How to navigate Atlas Volume, and Sample Images
>**Hardware requirements**
>- A mouse with left button, right button and scroll wheel  
>- A keyboard with numpad (not necessary but good to have)


1. Use mouse:
    - Mouse cursor inside left viewer to `scroll` along atlas AP-axis
    - Mouse cursor inside right viewer to `scroll` between sample images
    - Drag `AP slider` (left viewer, bottom) to go along AP-axis
    - Drag `ML slider` (left viewer, top) to change medio-lateral (ML) angle
    - Drag `DV slider` (left viewer, left) to change dorsal-ventral (DV) angle

2. Use keyboard:
    - `Numpad 2 or 8` to fine tune DV angle
    - `Numpad 4 or 6` to fine tune ML angle









