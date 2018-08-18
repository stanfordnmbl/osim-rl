---
title: Available models
permalink: /docs/models/
redirect_from: /docs/models/index.html
---

### Arm2DEnv

Simple 2D arm environment with 2 degrees of freedom and 6 muscles. The objective is to grab a ball randomly appearing in the space every 3 seconds.

| ---:| --- |
| **# of muscles** | 6 |
| **# degrees of freedom** | 2 |
| **reward** | negative distance from the requested point |

![3D arm environment](https://s3.amazonaws.com/osim-rl/videos/arm2d.gif)

<div class="note info">
  <h5>Toy model</h5>
  <p>Note that Arm2DEnv model is simplified and not physiologically accurate. We suggest using it only for testing purposes.</p>
</div>

### L2RunEnv

NIPS 2017 challenge model, where you are asked to build a controller to make a musculoskeletal model run as quickly as possible. Read more in the documents on the [NIPS 2017 challenge](/docs/nips2017/).

| ---:| --- |
| **# of muscles** | 18 |
| **# degrees of freedom** | 9 |
| **reward** | distance travelled in a simulation step |

![HUMAN environment](https://s3.amazonaws.com/osim-rl/videos/running.gif)

### ProstheticsEnv

NIPS 2018 challenge model, where you are asked to build a controller to make a musculoskeletal model with a prosthetic leg follow desired velocity vector changing in time. Read more in the documents on the [NIPS 2018 challenge](/docs/nips2018/).

| ---:| --- |
| **# of muscles** | 19 |
| **# degrees of freedom** | 14 |
| **reward** | negative distance from requested velocity |

![HUMAN environment](https://s3.amazonaws.com/osim-rl/images/prosthetic-leg.jpg)
