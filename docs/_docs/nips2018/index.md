---
title: About AI for prosthetics
permalink: /docs/nips2018/
redirect_from: /docs/nips2018/index.html
---

This repository contains software required for participation in the NIPS 2018 Challenge: AI for prosthetics. See more details about the challenge [**here**](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge).

<!--div class="note unreleased">
  <h5>Experimental</h5>
  <p>Current version of the NIPS 2018 challenge environment is experimental.</p>
</div-->

In this competition, you are tasked with developing a controller to enable a physiologically based human model with a prosthetic leg to walk in requested many directions with varying speeds. You are provided with a human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion.

<table style="background-color: #ffffff">
<caption align="bottom" style="padding-top: 0.3em; font-size: 0.8em">An amputee and <a href="https://simtk.org/projects/bkamputee_model">a below-knee prosthesis model in OpenSim</a>.</caption>
<tr><td><img src="https://s3.amazonaws.com/osim-rl/images/comparison.png" alt=""/></td></tr>
</table>

You are scored based on the adherence to the requested speed and direction of walking. You are given a parametrized training environment to help build your controllers. You are scored based on a final environment with unknown parameters.

## Why?

Recent advancements in material science and device technology have increased interest in creating prosthetics for improving human movement. Designing these devices, however, is difficult as it is costly and time-consuming to iterate through many designs. This is further complicated by the large variability in response among many individuals. One key reason for this is that our understanding of the interactions between humans and prostheses is not well-understood, which limits our ability to predict how a human will adapt his or her movement. Physics-based, biomechanical simulations are well-positioned to advance this field as it allows for many experiments to be run at low cost. Recent developments in using reinforcement learning techniques to train realistic, biomechanical models will be key to better understand the human-prosthesis interaction, which will help to accelerate development of this field.
