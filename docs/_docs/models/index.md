---
title: Available models
permalink: /docs/models/
redirect_from: /docs/models/index.html
---

### Arm2DEnv

Simple 2D arm environment with 2 degrees of freedom and 6 muscles. The objective is to grab a ball randomly appearing in the space every 3 seconds.

| ---:| --- |
| **# of muscles** | 2 |
| **# degrees of freedom** | 6 |
| **reward** | negative distance from the requested point |

![3D arm environment](https://s3.amazonaws.com/osim-rl/videos/arm2d.gif)

### L2RunEnv

NIPS 2017 challenge model, where you are asked to build a controler to make a musculoskeletal model run as quickly as possible. Read more in the documents on the [NIPS 2017 challenge](/docs/nips2017/).

| ---:| --- |
| **# of muscles** | 9 |
| **# degrees of freedom** | 18 |
| **reward** | distance travelled in a simulation step |

![HUMAN environment](https://s3.amazonaws.com/osim-rl/videos/running.gif)

### ProstheticEnv

NIPS 2018 challenge model, where you are asked to build a controler to make a musculoskeletal model with a prosthetic leg. Read more in the documents on the [NIPS 2018 challenge](/docs/nips2018/).

| ---:| --- |
| **# of muscles** | 14 |
| **# degrees of freedom** | 22 |
| **reward** | negative distance from requested velocity |

![HUMAN environment](https://s3.amazonaws.com/osim-rl/images/prosthetic-leg.jpg)
