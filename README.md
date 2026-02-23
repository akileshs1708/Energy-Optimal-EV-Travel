# Energy-Optimal Electric Vehicle Travel Using Extended Branch & Bound Methods

This project implements the methods from the paper **“Branch and Bound based methods to minimize the energy consumed by an electrical vehicle on long travels with slopes.”**  
It focuses on optimizing EV energy consumption over long, slope-varying routes using an enhanced Branch and Bound (B&B) algorithm supported by realistic electrical and mechanical vehicle models.

![2025-11-0511 00 44 PM](https://github.com/user-attachments/assets/90b2837f-ec3d-4464-9d3e-84bbf79cd5e5)
![21247ef5-0e3d-4ce1-bafe-bbb4889aeabe](https://github.com/user-attachments/assets/3f7b770b-7af5-4355-ad4d-d27286956ae9)

## Overview
The original B&B algorithm was developed for short (100 m), flat routes.  
This implementation extends the method to:
- Long-distance travel
- Routes with varying slopes
- Mixed-integer optimal control
- Realistic acceleration and deceleration phases

## Key Features
- Extended Branch & Bound search with variable time discretization
- Two new bounding heuristics for faster and more reliable pruning
- Slope-integrated electrical and mechanical EV dynamics
- Algorithm for handling slope transitions across route segments
- Acceleration constraints to ensure physically realistic solutions

## Outputs
- Optimal speed profile
- Optimal acceleration and braking phases
- Total energy consumption
- Energy usage per segment
- Visualizations generated through `visualize.py`

## Reference
This implementation is based on the paper:  
**“Branch and Bound based methods to minimize the energy consumed by an electrical vehicle on long travels with slopes.”**

## Credits
- Devkumar K 23PD06
- Akilesh S 23PD33
- Sabariesh Karthic A M 23PD34

