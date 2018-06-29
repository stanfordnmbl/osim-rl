---
title: Experimental data
---

Unlike in our NIPS 2017, now you can use publicly available experimental data to bootstrap the algorithm. Example running and walking kinematics can be found in many publicly available papers. For example we refer to *Schwartz, M. et al. (2018)* (citation details below).

You can find there [joint angles](https://s3.amazonaws.com/osim-rl/data/schwartz2008data/joint_angles.txt) and [EMG signals](https://s3.amazonaws.com/osim-rl/data/schwartz2008data/emg.txt). Data is represented as a function of time in a gait cycle (one step) at different speeds.

You can use it for supervised learning for bootstrapping your models (where for given kinematics you predict muscle activity).

    @article{schwartz2008effect,
      title={The effect of walking speed on the gait of typically developing children},
      author={Schwartz, Michael H and Rozumalski, Adam and Trost, Joyce P},
      journal={Journal of biomechanics},
      volume={41},
      number={8},
      pages={1639--1650},
      year={2008},
      publisher={Elsevier}
    }
