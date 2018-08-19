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
| **toes_l** | Left toes |
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
| **knee_r** | 1 | Right knee flexion |

Extra joints `back_0`, `mtp_l`, `subtalar_l` appear in the dictionary for consistency but they cannot move.

### Muscles

| *name* | *description* | *primary function(s)* |
| **abd_l** | Hip abductors (left) | Hip abduction (away from body's vertical midline) |
| **abd_r** | Hip abductors (right) | Hip abduction (away from body's vertical midline) |
| **add_l** | Hip adductors (left) | Hip adduction (toward body's vertical midline) |
| **add_r** | Hip adductors (right) | Hip adduction (toward body's vertical midline) |
| **bifemsh_l** | Short head of the biceps femoris (left) | Knee flexion |
| **bifemsh_r** | Short head of the biceps femoris (right) | Knee flexion |
| **gastroc_l** | Gastrocnemius (left) | Knee flexion and ankle extension (plantarflexion) |
| **glut_max_l** | Gluteus maximus (left) | Hip extension |
| **glut_max_r** | Gluteus maximums (left) | Hip extension |
| **hamstrings_l** | Biarticular hamstrings (left) | Hip extension and knee flexion |
| **hamstrings_r** | Biarticular hamstrings (right) | Hip extension and knee flexion |
| **iliopsoas_l** | iliopsoas (left) | Hip flexion |
| **iliopsoas_r** | iliopsoas (right) | Hip flexion |
| **rect_fem_l** | rectus femoris (left) | Hip flexion and knee extension |
| **rect_fem_r** | rectus femoris (right) | Hip flexion and knee extension |
| **soleus_l** | soleus (left) | Ankle extension (plantarflexion) |
| **tib_ant_l** | tibialis anterior (left) | Ankle flexion (dorsiflexion) |
| **vasti_l** | vasti (left) | Knee extension
| **vasti_r** | vasti (right) | Knee extension

Each muscle element is a dictionary with 4 elements

| ---:| --- |
| **activation** | Current activation |
| **fiber_force** | Current fiber force |
| **fiber_length** | Current fiber length |
| **fiber_velocity** | Current fiber velocity |


### Forces

| ---:| --- |
| **AnkleLimit_l** | Ankle ligament forces (left) |
| **AnkleLimit_r** | Ankle ligament forces (right) |
| **HipAddLimit_l** | Hip adduction/abduction ligament forces (left) |
| **HipAddLimit_r** | Hip adduction/abduction ligament forces (right) |
| **HipLimit_l** | Hip flexion/extension ligament forces (left) |
| **HipLimit_r** | Hip flexion/extension ligament forces (right) |
| **KneeLimit_l** | Knee flexion/extension ligament forces (left) |
| **KneeLimit_r** | Knee flexion/extension ligament forces (right) |
| **foot_l** | Ground reaction forces on the left foot. 6 values correspond to the 3 components (x,y,z) of the force and torque applied to the `foot_l` body. |
| **pros_foot_r_0** | Ground reaction forces on the prosthetic foot 6 values correspond to the 3 components (x,y,z) of the force and torque applied to the `pros_foot_r` body. |

For the difference between muscle forces in this item and `fiber_force` in muscles please refer to [Issue #163](https://github.com/stanfordnmbl/osim-rl/issues/163)

### Misc

| ---:| --- |
| **mass_center_pos** | Position of the center of mass `[x,y,z]` |
| **mass_center_vel** | Translational velocity of the center of mass `[x,y,z]` |
| **mass_center_acc** | Translational acceleration of the center of mass `[x,y,z]` |
