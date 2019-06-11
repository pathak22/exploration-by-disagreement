## Self-Supervised Exploration via Disagreement ##
#### [[Project Website]](https://pathak22.github.io/exploration-by-disagreement/) [[Demo Video]](https://youtu.be/POlrWt32_ec)

[Deepak Pathak*](https://people.eecs.berkeley.edu/~pathak/), [Dhiraj Gandhi*](http://www.cs.cmu.edu/~dgandhi/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg/)<br/>
(&#42; equal contribution)

UC Berkeley<br/>
CMU<br/>
Facebook AI Research

<a href="https://pathak22.github.io/exploration-by-disagreement/">
<img src="https://pathak22.github.io/exploration-by-disagreement/resources/method.jpg" width="600">
</img></a>

This is a TensorFlow based implementation for our [paper on self-supervised exploration via disagreement](https://pathak22.github.io/exploration-by-disagreement/). In this paper, we propose a formulation for exploration inspired by the work in active learning literature. Specifically, we train an ensemble of dynamics models and incentivize the agent to explore such that the disagreement of those ensembles is maximized. This allows the agent to learn skills by exploring in a self-supervised manner without any external reward. Notably, we further leverage the disagreement objective to optimize the agent's policy in a differentiable manner, without using reinforcement learning, which results in a sample-efficient exploration. We demonstrate the efficacy of this formulation across a variety of benchmark environments including stochastic-Atari, Mujoco, Unity and a real robotic arm. If you find this work useful in your research, please cite:

    @inproceedings{pathak19disagreement,
        Author = {Pathak, Deepak and
                  Gandhi, Dhiraj and Gupta, Abhinav},
        Title = {Self-Supervised Exploration via Disagreement},
        Booktitle = {ICML},
        Year = {2019}
    }

### Installation and Usage
The following command should train a pure exploration agent on Breakout with default experiment parameters.
```bash
python run.py
```
To use more than one gpu/machine, use MPI (e.g. `mpiexec -n 8 python run.py` should use 1024 parallel environments to collect experience instead of the default 128 on an 8 gpu machine).

### Other helpful pointers
- [Paper](https://pathak22.github.io/exploration-by-disagreement/resources/icml19.pdf)
- [Project Website](https://pathak22.github.io/exploration-by-disagreement/)
- [Demo Video](https://youtu.be/POlrWt32_ec)

### Acknowledgement

This repository is built off the publicly released code of [Large-Scale Study of Curiosity-driven Learning, ICLR 2019](https://github.com/openai/large-scale-curiosity).
