---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "DEFT"
date:   2024-02-21 
description: >- # Supports markdown
  DEFT: **D**ifferentiable Branched Discrete **E**lastic Rods for Modeling **F**urcated DLOs in Real-**T**ime
show-description: true

# Add page-specific mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: https://raw.githubusercontent.com/yich7045/DEFORM/DEFT/web_elements/demo_image.png
  height: 120
  width: 340
  alt: Random Landscape
  
authors:
  - name: Yizhou Chen
    email: yizhouch@umich.edu
  - name: Xiaoyue Wu
    email: wxyluna@umich.edu
  - name: Yeheng Zong
    email: yehengz@umich.edu
  - name: Anran Li
    email: anranli@umich.edu
  - name: Yuzhen Chen
    email: yuzhench@umich.edu
  - name: Julie Wu
    email: jwuxx@umich.edu
  - name: Bohao Zhang
    email: jimzhang@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

author-footnotes:
  All authors affiliated with the department of Mechanical Engineering and Department of Robotics of the University of Michigan, Ann Arbor.
  
  
links:
  - icon: arxiv
    icon-library: simpleicons
    text: Arxiv
    url: https://arxiv.org/abs/2406.05931
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/DEFT
    

# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

---

# Abstract

Autonomous wire harness assembly requires robots to manipulate complex branched cables with high precision and reliability.
A key challenge in automating this process is predicting how these flexible and branched structures behave under manipulation.
Without accurate predictions, it is difficult for robots to reliably plan or execute assembly operations.
While existing research has made progress in modeling single-threaded Deformable Linear Objects (DLOs), extending these approaches to Branched Deformable Linear Objects (BDLOs) presents fundamental challenges. 
The junction points in BDLOs create complex force interactions and strain propagation patterns that cannot be adequately captured by simply connecting multiple single-DLO models.
To address these challenges, this paper presents Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT), a novel framework that combines a differentiable physics-based model with a learning framework to: 1) accurately model BDLO dynamics, including dynamic propagation at junction points and grasping in the middle of a BDLO, 2) achieve efficient computation for real-time inference, and 3) enable planning to demonstrate dexterous BDLO manipulation.
A comprehensive series of real-world experiments demonstrates DEFT's efficacy in terms of accuracy, computational speed, and generalizability compared to state-of-the-art alternatives. 
<p align="center">
  <img src="https://raw.githubusercontent.com/yich7045/DEFORM/DEFT/web_elements/demo_image.png" class="img-responsive" alt="DEFORM model" style="width: 100%; height: auto;">

</p>
The figures above illustrate how DEFT can be used to autonomously perform a wire insertion task. 
**Left:** The system first plans a shape-matching motion, transitioning the BDLO from its initial configuration to the target shape (contoured with yellow), which serves as an intermediate waypoint. 
**Right:** Starting from the intermediate configuration, the system performs thread insertion, guiding the BDLO into the target hole while also matching the target shape. Notably, DEFT predicts the shape of the wire recursively without relying on ground truth or perception data at any point in the process.


---


# Method
<div markdown="1" class="content-block grey justify no-pre">
<p align="center">
  <img src="https://raw.githubusercontent.com/yich7045/DEFORM/DEFT/web_elements/DEFT_algorithm.png" class="img-responsive" alt="DEFORM overview" style="width: 140%; height: auto;">
</p>
Algorithm Overview of DEFT.
In the initialization stage, DEFT begins by separating the BDLO into a parent DLO and one or more children DLOs. 
Each DLO is discretized into vertices and represented as elastic rods. 
This setup allows DEFT to capture the geometric and physical properties required for dynamic simulation.
To improve computational efficiency, DEFT then predicts the dynamics of each branch in parallel. 
During this process, analytical gradients are provided to minimize potential energy, ensuring efficient and stable convergence.
Next, to address numerical errors, DEFT employs a GNN designed to learn the BDLO’s residual dynamics. 
By modeling discrepancies between simulated and observed behavior, the GNN refines predictions and enhances overall accuracy.
After integration, DEFT enforces constraints to enforce physical realism.
Inextensibility constraints are applied to each branch, while junction-level constraints ensure proper attachment at branch junctions. 
Additionally, edge orientation constraints enable the propagation of dynamics across these junctions.
Throughout the entire pipeline, all components remain fully differentiable, allowing for efficient parameter learning from real-world data.
</div>

---

# Modeling Results Visualization
<p align="center">
    <img src="https://raw.githubusercontent.com/yich7045/DEFORM/DEFT/web_elements/modeling_demo.png" class="img-responsive" alt="DEFORM overview" style="width: 140%; height: auto;">
</p>
<p align="center">
    <img src="https://raw.githubusercontent.com/yich7045/DEFORM/DEFT/web_elements/modeling_demo2.png" class="img-responsive" alt="DEFORM overview" style="width: 140%; height: auto;">
</p>
Visualization of the predicted trajectories for BDLO 1 under two manipulation scenarios, using DEFT, a DEFT ablation that leaves out the constraint described in Theorem 4, and Tree-LSTM. The ground-truth initial position of the vertices are colored in blue, the ground-truth final position of the vertices are colored in pink, and the gradient between these two colors is used to denote the ground truth location over time. 
The predicted vertices are colored as green circles (DEFT), orange circles (DEFT ablation), and light red circles (Tree-LSTM), respectively.
A gradient is used for these predictions to depict the evolution of time, starting from dark and going to light.
Note that the ground truth is only provided at t=0s and prediction is constructed until t=8s.
The prediction is performed recursively, without requiring additional ground-truth data or perception inputs throughout the entire process.

---

# Dataset
- For each BDLO, dynamic trajectory data is captured in real-world settings using a motion capture system operating at 100 Hz when robots grasp the BDLO’s ends. For details on dataset usage, please refer to DEFT_train.py.
- For BDLO 1 and BDLO 3, we record dynamic trajectory data when one robot grasps the middle of the BDLO while the other robot grasps one of its ends.
  
---

# Demo Video
<div class="fullwidth">
<video controls="" width="100%">
    <source src="https://raw.githubusercontent.com/yich7045/DEFORM/main/web_elements/RSS_2025_DEFT.mp4" type="video/mp4">
</video>
</div>
<div markdown="1" class="content-block grey justify">
  
---

# [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at University of Michigan - Ann Arbor.

```bibtex
@misc{chen2024differentiable,
      title={Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects}, 
      author={Yizhou Chen and Yiting Zhang and Zachary Brei and Tiancheng Zhang and Yuzhen Chen and Julie Wu and Ram Vasudevan},
      year={2024},
      eprint={2406.05931},
      archivePrefix={arXiv},
      primaryClass={id='cs.RO' full_name='Robotics' is_active=True alt_name=None in_archive='cs' is_general=False description='Roughly includes material in ACM Subject Class I.2.9.'}
}
```
</div>

---
