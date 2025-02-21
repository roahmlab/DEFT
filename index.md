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
  path: https://raw.githubusercontent.com/yich7045/DEFORM/blob/main/web_elements/DEFORM_model.jpg
  height: 100
  width: 256
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
    url: https://github.com/roahmlab/DEFORM
    

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
  <img src="https://raw.githubusercontent.com/yich7045/DEFORM/main/web_elements/DEFORM_model.jpg" class="img-responsive" alt="DEFORM model" style="width: 100%; height: auto;">

</p>
The figure shows DEFORM's predicted states (yellow) and the actual states (red) for a DLO over 4.5 seconds at 100 Hz. Note that the prediction is performed recursively, without requiring access to ground truth or perception during the process.

---


# Method
<div markdown="1" class="content-block grey justify no-pre">
DEFORM introduces a novel differentiable simulator as a physics prior for physics-informed learning to model DLOs in the real world. 
The following figure demonstrates the overview of DEFORM. Contributions of DEFORM are highlighted in green. 
a) DER models discretize DLOs into vertices, segment them into elastic rods, and model their dynamic propagation. 
DEFORM reformulates Discrete Elastic Rods(DER) into Differentiable DER (DDER) which describes how to compute gradients from the prediction loss, enabling efficient system identification and incorporation into deep learning pipelines.
b) To compensate for the error from DER's numerical integration, DEFORM introduces residual learning via DNNs.
c) 1 &rarr; 2: DER enforces inextensibility, but this does not satisfy classical conservation principles.  1 &rarr; 3: DEFORM enforces inextensibility with momentum conservation, which allows dynamic modeling while maintaining simulation stability.
<p align="center">
  <img src="https://raw.githubusercontent.com/yich7045/DEFORM/main/web_elements/DEFORM_Overview.png" class="img-responsive" alt="DEFORM overview" style="width: 100%; height: auto;">
</p>
</div>

---

# Dataset
<div markdown="1" class="content-block grey justify no-pre">
For each DLO, we collect 350 seconds of dynamic trajectory data in the real-world using the motion capture system at a frequency of 100 Hz. For dataset usage, please refer to train_DEFORM.py in [here](https://github.com/roahmlab/DEFORM).
</div>

---

# Demo Video
<div class="fullwidth">
<video controls="" width="100%">
    <source src="https://raw.githubusercontent.com/yich7045/DEFORM/main/web_elements/DEFORM_Demo.mp4" type="video/mp4">
</video>
</div>

<div markdown="1" class="content-block grey justify">

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
