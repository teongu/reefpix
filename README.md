# ReefPix: Pixel-Based Coral Reef Seabed Classification Tool

## Overview

ReefPix is an open-source tool for automated pixel-based classification of coral reef seabeds using multispectral satellite imagery. Developed as part of the study "Pixel-based satellite mapping for coral island seabed classification: application to the Maupiti island, French Polynesia", this tool combines machine learning (Random Forest) with segmentation techniques to map benthic habitats with up to 90% accuracy across 10 distinct classes (e.g., ReefFront, Lagoon, PatchReef).

**Key features:**

  1. Pixel-based classification with neighborhood contextual analysis.
  2. Segmentation smoothing (Felzenszwalb-Huttenlocher) to refine predictions into coherent geomorphic zones.
  3. Post-processing using expert-defined adjacency rules to correct misclassifications.
  4. Multi-resolution support (tested at 2m–10m resolution with Sentinel-2 and Pleiades imagery).

 ## Methodology

Input: Multispectral satellite images (RGB + NIR bands).

Pixel-based RF Classifier: Predicts seabed class for each pixel using surrounding neighborhood data.

Segmentation: Groups pixels into objects for smoother outputs.

Post-processing: Applies connectivity rules to enforce realistic class adjacencies.

## Quick Start

Clone the repo:

    git clone https://github.com/teongu/reefpix.git  
    cd reefpix  

Install dependencies:

    pip install -r requirements.txt  

Run the classification pipeline (see app/main.py for examples).

## Performances

| Resolution | Pixel-Based Accuracy | Smoothed Accuracy | Post-Processed Accuracy |
|------------|----------------------|-------------------|-------------------------|
| 10 m       | 90.7% ± 0.5%         | 86.9% ± 0.6%      | 87.0% ± 0.5%            |
| 5 m        | 91.2% ± 0.4%         | 89.3% ± 0.3%      | 89.4% ± 0.4%            |
| 3 m        | 91.1% ± 0.3%         | 90.2% ± 0.3%      | 90.3% ± 0.2%            |
| 2 m        | 90.0% ± 0.3%         | 88.8% ± 0.4%      | 88.9% ± 0.4%            |

## Acknowledgments

The GLADYS research team (https://www.gladys-littoral.org/) provided the data. The Maupiti satellite image comes from Pléiades satellite (Pléiades © CNES _ 2021, Distribution AIRBUS DS).
