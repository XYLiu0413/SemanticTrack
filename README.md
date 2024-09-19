## SemanticTrack
Radar detection systems often face challenges with ghost targets or false positives, especially when targets are in motion. To address this issue, this letter proposes a novel target clustering and track management method which integrates semantic information from radar point cloud. Unlike previous work, which has not effectively utilized velocity and Radar Cross Section (RCS) information provided by radar, our approach integrates this semantic information into the target detection and track management process, enhancing the radar detection performance. Due to the lack of annotations for ghost targets in existing open-source radar point cloud datasets, we collect a dedicated dataset to conduct experiments. The results demonstrate that our method achieves state-of-the-art performance compared to other methods in terms of object tracking accuracy and ghost suppression effectiveness.

### Code
- **main.py**: The main script to run the entire pipeline of our proposed method.
- **SemanticSceneRestriction.py**: Semantic Scene Restriction.
- **Pointnet2.py**: the Overall Semantic Segmentation model.
- **MOT.py**:  MOT and Track Management.
  
### Experimental Details
- **docs/ImplementationDetails.pdf**: This PDF document provides a comprehensive details of the implementation steps and specifics of our method as described in our paper. 

### Results
![Qualitative Results](docs/results.png)
Qualitative results of our method and the baselines. Note: The legend at the bottom of the image only applies to the results of our method (last line).
