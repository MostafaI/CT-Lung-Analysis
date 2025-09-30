# CT-Lung-Analysis

CT lung analysis pipeline: DICOM → NIfTI → lung segmentation → air-content visualization.

- **Input:** DICOM image series.
- **Output (same folder as input):**
  - `exampleCT.nii.gz` — converted CT volume (NIfTI)
  - `exampleCT_m.nii.gz` — binary lung mask
  - `exampleCT_air_content.png` — air-content map (quick QA image)

---

# 


## Usage
**Create environment using conda**
```
conda env create -f environment.yml
```

**Run the code**
```python
dicom_dir = "/path/to/DICOM_Study_or_Series"
run(dicom_dir)
```

**Expected console output examples:**
```
Converting DICOM to NIfTI...
Segmenting the lungs...
Analyzing the image...
Saved the output image in /path/to/Air_content.png
Program is complete, finished in 51 seconds
```
or :
```
NIfTI files are already there... skip.
Segmentation mask is already there... skip.
The output image (air_content) is already there... skip.
Program is complete, finished in 0 seconds
```


## Outputs

- **NIfTI CT:** `exampleCT.nii.gz`
- **Lung mask:** `exampleCT_m.nii.gz` (0/1)
- **Air content PNG:** `exampleCT_air_content.png`

> After running, **inspect the air-content image**. If it looks wrong, open the mask (`*_m.nii.gz`) in ITK-SNAP and check that the lungs are correctly segmented, Let Mostafa Know if you still need help.




## Contact
Questions: **mostafai@seas.upenn.edu**
