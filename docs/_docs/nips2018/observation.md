---
title: Observation dictionary
---

The observation at each step is a dictionary with following keys:

| ---:| --- |
| **body_pos** | Positions of body parts along (X,Y,Z) |
| **body_pos_rot** | Rotations of body parts along (X,Y,Z) |
| **body_vel** | Velocities of body parts along (X,Y,Z) |
| **body_vel_rot** | Rotational velocities of body parts along (X,Y,Z) |
| **body_acc** | Accelerations of body parts along (X,Y,Z) |
| **body_acc_rot** | Rotational accelerations of body parts along (X,Y,Z) |
| **joint_pos** | Positions of joints |
| **joint_vel** | Velocities of joints |
| **joint_acc** | Accelerations of joints |
| **muscles** | Forces acting on body parts |
| **forces** | Forces acting on body parts |
| **misc** | Forces acting on body parts |
| **markers** | Empty for this model |

Each element is a dictionary.

### Body parts

The model has following bodies

| ---:| --- |
| **calcn_l** | Left calcaneus |
| **talus_l** | Left talus |
| **tibia_l** | Left tibia |
| **toes_l** | Left teos |
| **femur_l** | Left femur |
| **femur_r** | Right femur |
| **head** | Head |
| **pelvis** | Pelvis |
| **torso** | Torso |
| **pros_foot_r** | Prosthetic foot |
| **pros_tibia_r** | Prosthetic tibia |

Each element is a vector. Elements corresponding to each transational position, velocity, and acceleration is order as `[x,y,z]`. Rotational position, velocity, and acceleration vectors correspond to rotations along X, Y, and Z.

### Joints

The model has following joints

| *name* | *length* | *description* |
| **ankle_l** | 1 | Left ankle flexion |
| **ankle_r** | 1 | Right ankle flexion |
| **back** | 1 | Back flexion |
| **ground_pelvis** | 6 | translation (x,y,z) and rotation along x, y, and z |
| **hip_l** | 3 | Hip flexion, abduction, and rotation |
| **hip_r** | 3 | Hip flexion, abduction, and rotation |
| **knee_l** | 1 | Left knee flexion |
| **knee_r** | 1 |Right knee flexion |

Extra joints `back_0`, `mtp_l`, `subtalar_l` appear in the dictionary for consistency but they cannot move.

### Muscles

(For the moment, please refer to [this](https://cdn-images-1.medium.com/max/800/1*o5o1M7__pT9lOL5ez2ehMw.png))

| ---:| --- |
| **abd_l** |  |
| **abd_r** |  |
| **add_l** |  |
| **add_r** |  |
| **bifemsh_l** |  |
| **bifemsh_r** |  |
| **gastroc_l** |  |
| **glut_max_l** |  |
| **glut_max_r** |  |
| **hamstrings_l** |  |
| **hamstrings_r** |  |
| **iliopsoas_l** |  |
| **iliopsoas_r** |  |
| **rect_fem_l** |  |
| **rect_fem_r** |  |
| **soleus_l** |  |
| **tib_ant_l** |  |
| **vastil_l** |  |
| **vastil_r** |  |

Each muscle element is a dictionary with 4 elements

| ---:| --- |
| **activation** | Current activation of the given muscle |
| **fiber_force** | Current fiber force |
| **fiber_length** | Current fiber length |
| **fiber_velocity** | Current fiber velocity |


### Forces

| ---:| --- |
| **AnkleLimit_l** | Ankle limit forces |
| **AnkleLimit_r** |  |
| **HipAddLimit_l** |  |
| **HipAddLimit_r** |  |
| **HipLimit_l** |  |
| **HipLimit_r** |  |
| **KneeLimit_l** |  |
| **KneeLimit_r** |  |
| **foot_l** | Ground reaction forces on the left foot |
| **pros_foot_r_0** | Ground reaction forces on the prosthetic foot |

Note that in the forces dictionary, forces corresponding to muscles are redundant with `fiber_force` in muscles dictionaries (they are not listed above, but the keys appear in the dictionary).

### Misc

| ---:| --- |
| **mass_center_pos** | Position of the center of mass `[x,y,z]` |
| **mass_center_vel** | Translational velocity of the center of mass `[x,y,z]` |
| **mass_center_acc** | Translational acceleration of the center of mass `[x,y,z]` |
