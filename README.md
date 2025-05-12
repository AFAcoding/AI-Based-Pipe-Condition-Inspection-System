# AI-Based Pipe Condition Inspection System

## Project Overview

This project explores the feasibility of integrating an AI-based inspection module to automate the analysis of pipe conditions from inspection videos. The system's core functionalities include detecting and quantifying water levels, contamination, and blockages. The study also examines expanding capabilities such as measuring pipe diameter and estimating camera position and orientation.

---

## Objective

- Evaluate the performance of AI models for detecting pipe conditions from videos.
- Explore additional analysis like pipe diameter estimation and camera positioning.
- Assess edge device compatibility in terms of speed and resource efficiency.
- Review the quality and diversity of available labeled data.

---

## Background

The AI module automates video analysis to identify:
- **Water Levels**
- **Contamination**
- **Blockages**

### Additional Research Areas:
- **Pipe Diameter Measurement**
- **Camera Position & Orientation Estimation**

---

## Scope & Focus

### Core Functionalities:
- Water level, contamination, and blockage detection using computer vision.

### Extended Capabilities:
- Feasibility of measuring pipe diameter.
- Estimating the camera's spatial orientation.

### Technical Evaluation:
- Edge device performance.
- Labeled data quality and model generalizability.

---

## Methodology

1. Collect inspection video data.
2. Preprocess data and generate annotations.
3. Train object detection models (YOLO.).
4. Evaluate detection accuracy (mAP, precision, recall).
5. Deploy prototype to edge devices for benchmarking.

---

## Deliverables

- **Feasibility Study Report**
- **AI Prototype System**
- **Performance Benchmarks**
- **Future Enhancement Recommendations**

---

## Technology Stack

- **Frameworks**: PyTorch, YOLOv8, OpenCV
- **Languages**: Python
- **Tools**: FFmpeg, sklearn, YAML, shutil

---
## Related Work

This feasibility study is informed by a review of state-of-the-art methods used in automated pipe defect detection. Key prior work includes:

- **Halfawy & Hengmeechai (2014)**  
  [Automated defect detection using HOG + SVM](https://doi.org/10.1016/j.autcon.2013.10.012)  
  - High accuracy (~90%) on root defects using histogram of oriented gradients and SVM  
  - Limited dataset (291 images) and poor performance on other defect types  
  - Slow processing speed

- **Zhang et al. (2023)**  
  [YOLOv4 with SPP and DIoU Loss](https://doi.org/10.3390/app13074589)  
  - Best performing approach: mAP of 92.3%  
  - Incorporates spatial pyramid pooling for scale invariance and better context fusion  
  - Uses Distance-IoU loss and cosine annealing learning rate schedule  
  - Slightly slower than YOLOv3/v7, but more accurate and stable

- **Li et al. (2021)**  
  [Deep learning with local and global feature fusion](https://doi.org/10.1016/j.autcon.2021.103823)  
  - Combines global and local visual cues for better defect classification  
  - Emphasizes comprehensive feature representation for improved detection

- **Li et al. (2019)**  
  [CNN with hierarchical classification](https://doi.org/10.1016/j.autcon.2019.01.017)  
  - Addresses class imbalance via oversampling  
  - Achieves 83.2% accuracy in high-level defect detection  
  - Room for improvement in fine-grained (low-level) classifications

- **Yang & Su (2009)**  
  [Sewer defect segmentation](https://doi.org/10.1016/j.eswa.2008.02.006)  
  - Explores early methods for ideal morphology segmentation of defects  
  - Lays groundwork for image processing-based diagnosis techniques

- **Zhou et al. (2025)**  
  [Multi-defect detection in tunnels using deep learning](https://doi.org/10.1016/j.engappai.2025.110035)  
  - Fast and effective deep learning approach  
  - Relevant methodology for multiple defect detection transfer to sewer context

- **Kumar et al. (2018)**  
  [Binary classification for sewer defects with CNNs](https://doi.org/10.1016/j.autcon.2018.03.028)  
  - Separate binary CNNs for each defect type, ~90% accuracy  
  - Dataset limited to circular clay/concrete/iron pipes only  
  - Poor generalization to unseen defect combinations

- **Bahnsen et al. (2023)**  
  [3D pipe interior measurement using RANSAC](https://doi.org/10.1016/j.autcon.2023.104864)  
  - Measures pipe diameter via RANSAC-based cylinder fitting on 3D scans  
  - Achieved ±20 mm accuracy on real-world pipes (150–1100 mm diameter)  
  - Applicable for renovation/geometry extraction use-cases

- **Kolesnik & Baratoff**  
  - Circle detection using Hough transforms from monocular images  
  - Distance estimation through known pipe diameter and robot travel metrics

- **Obstruction Detection (2020)**  
  [arXiv:2002.01284](https://doi.org/10.48550/arXiv.2002.01284)  
  - Uses custom CNNs and OpenCV for obstruction level estimation  
  - Includes best-frame selection and efficient defect classification

---

**Summary**  
YOLO-based models offer the best trade-off between detection accuracy and inference speed. For interior measurements and water level estimation, computer vision techniques—particularly CNNs and 3D reconstruction—are widely used. Complementary methods such as segmentation, HOG features, and noise filtering further enhance performance across defect detection pipelines.


