###NOTES FROM REDDIT FANATICS:
"""
If you want to generate unbiased estimates of the probabilities of various outcomes, GANs are decidedly the wrong choice of probabilistic model. Yes, the samples will look realistic, but since GANs are susceptible to mode collapse, it's likely that the predictions are going to be biased towards a subset of modes which the generator has learned to model. And when you marginalize over the latent distribution, you're going to end up with blurry estimates no matter what, so the fact that the individual samples are sharp is basically useless (aside from deceiving people who don't understand the nuances of various generative models). In reference to the adversarial losses, they write:

>When used alone, these losses lead to accuracy on par with Eulerian persistence

I assume they don't define Eulerian persistence because, if they did, it would make clear that the GANs aren't doing shit (see https://rainymotion.readthedocs.io/en/latest/models.html#the-eulerian-persistence). Only once they add an l1 prediction loss do they get models with any predictive value. Yet despite the l1 loss doing the heavy lifting and the GAN loss doing nothing for predictive performance, they have the audacity to refer to the l1 loss as the 'regularization' term.

Their evaluation relies on meteorologists' assessments of the GAN forecasts vs the forecasts of other models. They write:

>Meteorologists were not swayed by the visual realism of the predictions

Then a couple sentences later they write that meteorologists described the GAN samples as having "much higher detail compared to what [expert meteorologists] are used to at the moment". Sounds like they were pretty clearly swayed (deceived) by the high frequency components which are visible in GAN samples but not in 'blurry' marginal estimates.

Seems like just a bunch of garbage dressed up with GANs to me. I have no qualifications to critique these kinds of nature-published-thousand-author papers that deepmind pumps out, so take my comments with a grain of salt. Happy to be proven wrong.

###RESPONSE TO ABOVE^
#You raise a lot of valid points, but I don't think it follows that the approach is garbage.

The GAN is basically just hallucinating plausible details on top of the L1 prediction, but the fact is, this still lead to a higher predictive skill and value! Is the method really garbage if it has higher predictive performance on multiple metrics than other leading deep networks and statistical baselines?

Furthermore, there is a ton of research into avoiding GAN mode-dropping that can be integrated into this baseline approach. That seems like a pretty promising way to gain even more performance!

I also think marginalizing over the latent distribution is very promising (would love to have seen an analysis of this in the paper!). Yes, of course that will be blurrier than single estimates, but the resolution of DGMR is 1x1 kilometer vs the effective resolution of UNet being 32km and Axial Attention being 16 km (at T+90 min). There's quite a lot of room to average predictions and still outperform the other methods.

On the predictive value side: the meteorologists weren't solely assessing DGMR positively because it had higher resolution, they also noted that the baselines were implausible and that DGMR had other benefits next to the high frequency details:

In the phase 2 interviews, PySTEPS was described as “being too developmental which would be misleading”, that is, as having many “positional errors” and “much higher intensity compared with reality”. The axial attention model was described as “too bland”, that is, as being “blocky” and “unrealistic”, but had “good spatial extent”. Meteorologists described DGMR as having the “best envelope”, “representing the risk best”, as having “much higher detail compared to what [expert meteorologists] are used to at the moment”, and as capturing “both the size of convection cells and intensity the best”.

These are experts in their field that definitely know what to look for when evaluating a forecasting model. I think characterizing them all as fools who are deceived by the higher number of pixels is not fair.
"""

# DeepMind Research

This repository contains implementations and illustrative code to accompany
DeepMind publications. Along with publishing papers to accompany research
conducted at DeepMind, we release open-source
[environments](https://deepmind.com/research/open-source/open-source-environments/),
[data sets](https://deepmind.com/research/open-source/open-source-datasets/),
and [code](https://deepmind.com/research/open-source/open-source-code/) to
enable the broader research community to engage with our work and build upon it,
with the ultimate goal of accelerating scientific progress to benefit society.
For example, you can build on our implementations of the
[Deep Q-Network](https://github.com/deepmind/dqn) or
[Differential Neural Computer](https://github.com/deepmind/dnc), or experiment
in the same environments we use for our research, such as
[DeepMind Lab](https://github.com/deepmind/lab) or
[StarCraft II](https://github.com/deepmind/pysc2).

If you enjoy building tools, environments, software libraries, and other
infrastructure of the kind listed below, you can view open positions to work in
related areas on our [careers page](https://deepmind.com/careers/).

For a full list of our publications, please see
https://deepmind.com/research/publications/


## Projects

*   [Skilful precipitation nowcasting using deep generative models of radar](nowcasting), Nature 2021
*   [Compute-Aided Design as Language](cadl)
*   [Encoders and ensembles for continual learning](continual_learning)
*   [Towards mental time travel: a hierarchical memory for reinforcement learning agents](hierarchical_transformer_memory)
*   [Perceiver IO: A General Architecture for Structured Inputs & Outputs](perceiver)
*   [Solving Mixed Integer Programs Using Neural Networks](neural_mip_solving)
*   [A Realistic Simulation Framework for Learning with Label Noise](noisy_label)
*   [Rapid Task-Solving in Novel Environments](rapid_task_solving), ICLR 2021
*   [WikiGraphs: A Wikipedia - Knowledge Graph Paired Dataset](wikigraphs), TextGraphs 2021
*   [Behavior Priors for Efficient Reinforcement Learning](box_arrangement)
*   [Learning Mesh-Based Simulation with Graph Networks](meshgraphnets), ICLR 2021
*   [Open Graph Benchmark - Large-Scale Challenge (OGB-LSC)](ogb_lsc)
*   [Synthetic Returns for Long-Term Credit Assignment](synthetic_returns)
*   [A Deep Learning Approach for Characterizing Major Galaxy Mergers](galaxy_mergers)
*   [Better, Faster Fermionic Neural Networks](kfac_ferminet_alpha) (KFAC implementation)
*   [Object-based attention for spatio-temporal reasoning](object_attention_for_reasoning)
*   [Effective gene expression prediction from sequence by integrating long-range interactions](enformer)
*   [Satore: First-order logic saturation with atom rewriting](satore)
*   [Characterizing signal propagation to close the performance gap in unnormalized ResNets](nfnets), ICLR 2021
*   [Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](adversarial_robustness)
*   [Learning rich touch representations through cross-modal self-supervision](cmtouch), CoRL 2020
*   [Functional Regularisation for Continual Learning](functional_regularisation_for_continual_learning), ICLR 2020
*   [Self-Supervised MultiModal Versatile Networks](mmv), NeurIPS 2020
*   [ODE-GAN: Training GANs by Solving Ordinary Differential Equations](ode_gan), NeurIPS 2020
*   [Algorithms for Causal Reasoning in Probability Trees](causal_reasoning)
*   [Gated Linear Networks](gated_linear_networks), NeurIPS 2020
*   [Value-driven Hindsight Modelling](himo), NeurIPS 2020
*   [Targeted free energy estimation via learned mappings](learned_free_energy_estimation), Journal of Chemical Physics 2020
*   [Learning to Simulate Complex Physics with Graph Networks](learning_to_simulate), ICML 2020
*   [Physically Embedded Planning Problems](physics_planning_games)
*   [PolyGen: PolyGen: An Autoregressive Generative Model of 3D Meshes](polygen), ICML 2020
*   [Bootstrap Your Own Latent](byol)
*   [Catch & Carry: Reusable Neural Controllers for Vision-Guided Whole-Body Tasks](catch_carry), SIGGRAPH 2020
*   [MEMO: A Deep Network For Flexible Combination Of Episodic Memories](memo), ICLR 2020
*   [RL Unplugged: Benchmarks for Offline Reinforcement Learning](rl_unplugged)
*   [Disentangling by Subspace Diffusion (GEOMANCER)](geomancer), NeurIPS 2020
*   [What can I do here? A theory of affordances in reinforcement learning](affordances_theory), ICML 2020
*   [Scaling data-driven robotics with reward sketching and batch reinforcement learning](sketchy), RSS 2020
*   [Path-Specific Counterfactual Fairness](counterfactual_fairness), AAAI 2019
*   [The Option Keyboard: Combining Skills in Reinforcement Learning](option_keyboard), NeurIPS 2019
*   [VISR - Fast Task Inference with Variational Intrinsic Successor Features](visr), ICLR 2020
*   [Unveiling the predictive power of static structure in glassy systems](glassy_dynamics), Nature Physics 2020
*   [Multi-Object Representation Learning with Iterative Variational Inference (IODINE)](iodine)
*   [AlphaFold CASP13](alphafold_casp13), Nature 2020
*   [Unrestricted Adversarial Challenge](unrestricted_advx)
*   [Hierarchical Probabilistic U-Net (HPU-Net)](hierarchical_probabilistic_unet)
*   [Training Language GANs from Scratch](scratchgan), NeurIPS 2019
*   [Temporal Value Transport](tvt), Nature Communications 2019
*   [Continual Unsupervised Representation Learning (CURL)](curl), NeurIPS 2019
*   [Unsupervised Learning of Object Keypoints (Transporter)](transporter), NeurIPS 2019
*   [BigBiGAN](bigbigan), NeurIPS 2019
*   [Deep Compressed Sensing](cs_gan), ICML 2019
*   [Side Effects Penalties](side_effects_penalties)
*   [PrediNet Architecture and Relations Game Datasets](PrediNet)
*   [Unsupervised Adversarial Training](unsupervised_adversarial_training), NeurIPS 2019
*   [Graph Matching Networks for Learning the Similarity of Graph Structured
    Objects](graph_matching_networks), ICML 2019
*   [REGAL: Transfer Learning for Fast Optimization of Computation Graphs](regal)
*   [Deep Ensembles: A Loss Landscape Perspective](ensemble_loss_landscape)
*   [Powerpropagation](powerpropagation)



## Disclaimer

*This is not an official Google product.*
