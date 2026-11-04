# Autonomous Galaxy Detection Pipeline

## Overview
An algorithmic computer vision pipeline designed to identify, isolate, and bound galaxy structures within deep-sky astronomical imagery. This project focuses on automating astronomical feature extraction from open-source sky survey datasets.

## Project Architecture & Methodology
- **Concept & Pipeline Design:** Saksham Sharma
- **Execution & Code Generation:** Designed workflow and logic, engineered prompts using ChatGPT (Python/OpenCV/NumPy).
- **Primary Data Sources:** Sloan Digital Sky Survey (SDSS) / Deep-Sky Open Archives.

## Data Pipeline
1. **Ingestion:** Loading deep-sky image arrays into Python.
2. **Pre-processing:** Grayscale conversion, Gaussian blurring for stellar noise reduction, and adaptive intensity thresholding.
3. **Contour Analysis:** Isolating contiguous pixel clusters corresponding to galaxian morphological profiles.
4. **Bounding Box Generation:** Drawing target coordinates and exporting classified objects.

## How to Run
```bash
# Clone this repository
git clone [https://github.com/sakshamsharma902/Galaxy-detector-pipeline.git](https://github.com/sakshamsharma902/Galaxy-detector-pipeline.git)
cd Galaxy-detector-pipeline

# Install required dependencies
pip install opencv-python numpy matplotlib

# Run the detection script
python detect_galaxies.py --input sample_sky.jpg
