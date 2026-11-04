import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_galaxy_image(image_path):
    """
    Automated Deep-Sky Galaxy Detection & Morphological Classification Pipeline
    Designed for astronomical feature extraction.
    """
    # 1. Ingestion: Load deep-sky image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Pre-processing: Gaussian blur to suppress stellar background noise
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    # 3. Adaptive Thresholding to isolate high-intensity galaxian cores
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 4. Contour Analysis: Find contiguous pixel clusters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Detected {len(contours)} potential galaxian candidate regions.")

    # 5. Bounding Box Generation & Classification
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAYBGR)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 50:  # Filter out minor star points
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Simple heuristic aspect-ratio check for morphological categorization
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                gal_type = "Elliptical"
                color = (0, 255, 0) # Green
            elif aspect_ratio > 1.2:
                gal_type = "Spiral"
                color = (255, 0, 0) # Blue
            else:
                gal_type = "Irregular"
                color = (0, 0, 255) # Red

            # Draw bounding box and label
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_img, f"{gal_type}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save output result
    cv2.imwrite("classified_galaxies_output.jpg", output_img)
    print("Pipeline execution complete. Result saved as 'classified_galaxies_output.jpg'.")

if __name__ == "__main__":
    # Example execution call
    process_galaxy_image("sample_sky.jpg")
              
