# Edge Classification System

This repository contains a modular, research-friendly implementation of an LSD-based junction and straight-edge detection pipeline.

Further, after junctions are detected, they are classified by their shape and used to classify the edges they are on into the following classes: Obstruction, Surface, Reflectance, and Unclassified.

Obstruction edges are those pertaining to objects occluding each other. 
Surface edges are those pertaining to the geometry of an object (i.e. the edges perceived when an object bends due to its shape). 
Reflectance edges are those created by changes in color. Unclassified edges are those which could not be places in any of the aforementioned classes.

The project is split into multiple modules:

- `models/`: Core data structures (`Line`, `Junction`, `StraightEdge`)
- `detectors/`: Logic for detecting lines, junctions, and branches
- `utils/`: Geometry computations, image I/O, clustering utilities
- `visualization/`: Drawing and exporting results
- `config.py`: Tunable parameters
- `main.py`: Entry point

## Usage

python main.py

Input images must be placed inside: selected/
BY DEFAULT, ONLY .png IMAGES ARE PROCESSED. This can be modified by changing 'selected/*.png' to the desired extension.

Results will be saved into: output/

This project is prepared for Git and structured as a Python package.

## Background

Adelson's research on the mammalian visual system is used as the main source for edge and junction classification, while junction detection is sourced from multiple IT papers. The audience is advised to check the available documentation in this repository and its referenced papers to understand the mathematical, theoretical, and statistical bases for this project.

## AI USAGE ADVISORY

The original code for this project was re-strustured with the help of AI. However, the original code and how it was used remains unchanged. This helped a single author re-factor and explain their code in a better way for GitHub audiences.
If any audience member desires to see the original code, which was all contained in a single .py file, said file is also in this project with the name 'edge_classification_original.py'. This file is free from any AI influence, but it might be harder to read. This file by itself can also produce the same output as the overall modular project.
