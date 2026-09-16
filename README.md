CancerAIMicrorobot

CancerAIMicrorobot is an experimental software project exploring how artificial intelligence could conceptually guide microscopic robots in identifying cancer cells while avoiding healthy cells.

Why I Started This Project

In late July 2026, I was diagnosed with cancer. As I continue battling it, I decided to take something I already care deeply about—computer science and artificial intelligence—and use it to explore an idea directly connected to what I’m going through.

The question behind this project is simple:

Could an AI system help a microscopic robot distinguish between cancerous and healthy cells, make decisions under uncertainty, and selectively target cancer cells while leaving healthy tissue alone?

This repository does not attempt to build a real medical device. Instead, I’m building a software simulation of how such a system might work.

The long-term goal is to create a visual simulation that begins with an injection into the human body, follows simulated microrobots through the bloodstream toward a tumor, and then zooms down to the cellular level. There, each robot will encounter cells, analyze simulated sensor data, estimate whether a cell is cancerous, and decide whether to target it, leave it alone, or take no action because the evidence is uncertain.

Current System

The first version implements the basic decision pipeline:

Simulated Cell
      |
      v
Simulated Sensor Measurements
      |
      v
AI / Classification System
      |
      v
Cancer Probability
      |
      v
Decision & Safety Layer
      |
      +---- PASS
      |
      +---- UNCERTAIN
      |
      +---- TARGET

The simulation deliberately separates the true identity of a cell from what the robot is allowed to observe. The classifier therefore cannot simply “know” whether a cell is cancerous.

Instead, it must make a decision using simulated measurements.

This allows the project to eventually explore problems such as:

* False positives
* False negatives
* Sensor noise
* Classification uncertainty
* Confidence thresholds
* Protecting healthy cells
* Detecting difficult cancer cells
* AI model comparison
* Explainable decisions

Project Roadmap

The project is being developed incrementally.

v0.1 — Decision System

* Simulated cells
* Simulated microrobot
* Cancer probability calculation
* TARGET / PASS / UNCERTAIN decisions
* Explainable feature contributions

v0.2 — Large-Scale Simulation

* Synthetic cell populations
* Sensor noise
* Cancer and healthy cell variation
* False-positive and false-negative measurement
* Sensitivity and specificity analysis
* Automated testing

v0.3 — Machine Learning

* Train classifiers on synthetic datasets
* Compare multiple ML approaches
* Evaluate model performance
* Experiment with safety thresholds

v0.4 — Microrobot Environment

* 2D tissue environment
* Robot navigation
* Cell encounters
* Tumor regions
* Multiple simulated robots

v0.5 — Visual Demonstration

* Human body visualization
* Injection sequence
* Bloodstream navigation
* Tumor localization
* Cellular-level zoom
* Real-time AI decisions
* Healthy-cell avoidance
* Simulated cancer-cell targeting

Important Disclaimer

This project is an educational and experimental software simulation.

It is not a medical device, treatment, clinical decision system, or validated method for detecting or treating cancer.

The current biomarkers, sensor measurements, classifier weights, robot behavior, and biological environment are simulated and should not be interpreted as medically validated parameters.

The project explores computer science, machine learning, robotics concepts, visualization, and decision-making under uncertainty.

Motivation

Cancer changed what I expected 2026 to look like.

Rather than allowing that experience to exist completely separately from my interests in computer science and AI, I wanted to build something from it.

I don’t know where this project will ultimately lead. For now, the goal is to learn, experiment, build, and explore what AI-driven systems might someday be capable of.