---
title: Frequently Asked Questions
---

**I'm getting 'version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference' error**

If you are getting this error:

    ImportError: /opensim-rl/lib/python2.7/site-packages/opensim/libSimTKcommon.so.3.6:
      symbol _ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE, version
      GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference

Try `conda install libgcc`.

**Can I use languages other than python?**

Yes, you just need to set up your own python grader and interact with it
[https://github.com/kidzik/osim-rl-grader](https://github.com/kidzik/osim-rl-grader). Find more details here [OpenAI http client](https://github.com/openai/gym-http-api)

**Some libraries are missing. What is required to run the environment?**

Most of the libraries by default exist in major distributions of operating systems or are automatically downloaded by the conda environment. Yet, sometimes things are still missing. The minimal set of dependencies under Linux can be installed with

    sudo apt install libquadmath0 libglu1-mesa libglu1-mesa-dev libsm6 libxi-dev libxmu-dev liblapack-dev

Please, try to find equivalent libraries for your OS and let us know -- we will put them here.

**Why there are no energy constraints?**

Please refer to the [issue #34](https://github.com/stanfordnmbl/osim-rl/issues/34).

**I have some memory leaks, what can I do?**

Please refer to
[issue #10](https://github.com/stanfordnmbl/osim-rl/issues/10)
and to
[issue #58](https://github.com/stanfordnmbl/osim-rl/issues/58)

**I see only python3 environment for Linux. How to install Windows environment?**

Please refer to
[issue #29](https://github.com/stanfordnmbl/osim-rl/issues/29)

**How to visualize observations when running simulations on the server?**

Please refer to
[issue #59](https://github.com/stanfordnmbl/osim-rl/issues/59)

**I still have more questions, how can I contact you?**

For questions related to the challenge please use [the challenge forum](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge/topics).
For issues and problems related to installation process or to the implementation of the simulation environment feel free to create an [issue on GitHub](https://github.com/stanfordnmbl/osim-rl/issues).

