# CT-Lung-Analysis

CT lung analysis pipeline: DICOM → NIfTI → lung segmentation → air-content visualization.

- **Input:** DICOM image series.
- **Output (DICOM__Analyzed):**
  - `exampleCT.nii.gz` — converted CT volume (NIfTI)
  - `exampleCT_m.nii.gz` — binary lung mask
  - `exampleCT_air_content.png` — air-content map
  - `exampleCT_inter.png` — inter-regional heterogeneity plot
  - `exampleCT_intra.png` — intra-regional heterogeneity plot
  - `exampleCT.csv` — inter-regional and intra-regional heterogeneities values saved
---

## Prebuilt Executable
Ongoing 


## Usage

**Run the code**
```python
dicom_dir = "/path/to/DICOM_Study_or_Series"
run(dicom_dir)
```

**Expected console output examples:**
```
Sample output:
    Converting Dicom to NIFT..
    Segmenting the lungs..
    Getting the air content map..
    Analyzing the image..
    Getting the heterogeneity maps..
    Program is complete, finished in 78 seconds
```
or :
```
NIFTI files are already there..skip..
Segmentation mask is already there..skip..
The air content image is already there..skip..
The heterogeneity csv file is already there..skip..
Program is complete, finished in 0 seconds
```

> After running, **inspect the air-content image**. If it looks wrong, open the mask (`*_m.nii.gz`) in ITK-SNAP and check that the lungs are correctly segmented, Let Mostafa Know if you still need help.


## Contact
Questions: **mostafai@seas.upenn.edu**
