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

