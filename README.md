# SemanticTrack: A Semantic Fusion Method for Radar Point Cloud Ghost Suppression


## Table of Contents
- [Introduction](#introduction)
- [Code Overview](#code-overview)
- [Experimental Details](#experimental-details)
- [Results](#results)
- [Environment](#Environment)
- [Data Availability](#data-availability)

## Introduction
SemanticTrack is a novel target clustering and track management method that integrates semantic information from radar point clouds. We are the first to fully exploit these features by training ghost targets as a distinct class. Due to the lack of ghost target annotations in existing open-source datasets, we collected a dedicated dataset for our experiments. Results show that our method improves radar-based perception systems, enhancing tracking and ghost suppression in complex environments.

## Code Overview

This repository contains the following key components:

- **main.py**: The main script that runs the entire pipeline of our proposed semantic fusion method.
- **SemanticSceneRestriction.py**: Semantic Scene Restriction. Fully utilize the spatial and semantic information provided by radar to form target point cloud groups.
- **Pointnet2.py**: Overall Semantic Segmentation model. infer the semantic labels of the target point cloud groups.
- **MOT.py**: MOT and Fusion Management. Target identification and ghost suppression are made according to the semantic labels and life cycles of all tracks.

## Experimental Details

Detailed implementation steps and specifics of our method can be found in the following document:

- **[docs/ImplementationDetails.pdf](docs/ImplementationDetails.pdf)**: This PDF outlines the comprehensive details of the experimental facilities.

## Results

Below are the qualitative results showcasing the effectiveness of our method in comparison to baseline methods. The legend at the bottom of the image applies only to the results of our method (last row).

![Qualitative Results](docs/results.png)

## Environment

Create a new Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```
## Data Availability

Currently, the data used in this project is subject to restrictions related to the collection site. As such, we are unable to release the dataset at this time. We are in the process of obtaining permission from the experimental site provider, and once the necessary approvals are granted, we will release the dataset as open source. 


