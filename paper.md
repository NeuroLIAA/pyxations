---
title: "Pyxations: organizing, parsing, and analysing eye-movement data using Python"
tags: [Python, eye tracking, data analysis, gaze, neuroscience]
authors:
  - name: "Gonzalo Ruarte"
    affiliation: 1
  - name: "Gustavo Juantorena"
    affiliation: 1
  - name: "Joaquín González"
    affiliation: "1, 3"
  - name: "Juan Esteban Kamienkowski"
    affiliation: "1, 2"

affiliations:
  - name: "Laboratorio de Inteligencia Artificial Aplicada, Instituto de Ciencias de la Computación, Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires - CONICET, Argentina"
    index: 1
  - name: "Maestría de Explotación de Datos y Descubrimiento del Conocimiento, FCEyN-FI, UBA, Argentina"
    index: 2
  - name: "Departamento de Física, Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires"
    index: 3

date: "2026-07-28"
bibliography: paper.bib
---

# Summary

Pyxations is a Python-based toolbox designed to unify the organization, parsing, and analysis of eye-movement data. It supports multiple common recording formats and detection algorithms, and offers integrated tools for preprocessing, visualization, and downstream analysis. Pyxations converts sample-level recordings from its supported eye trackers to the eye-tracking physiological format defined by BIDS and writes processed samples and detected eye movements to a linked, validator-tested BIDS Derivatives dataset using the general derivatives conventions and physiological recording file types. By standardizing these steps, Pyxations facilitates reproducibility and makes it easier to compare results across tasks, devices, and studies.

# Statement of Need

Looking at someone’s eyes, searching for keys, and reading are active processes in which eye movements play a crucial role [@rayner1998; @wade2005; @land2009looking; @holmqvist2011eye]. Eye-tracking recordings can describe not only fixations and saccades, but also smooth pursuit, microsaccades, vergence, blinks, and changes in pupil size. Blink timing and pupil dynamics can be scientifically informative in their own right and should therefore be retained even when an analysis primarily concerns gaze position [@zhang2026pupeyes]. Common remote laboratory eye-trackers use one or more cameras, typically with infrared illumination, to image the pupil and corneal reflection; pupil--corneal-reflection methods then estimate where the eyes are directed. Webcam-based systems generally operate without controlled infrared illumination and instead estimate gaze from eye and face images using computer-vision or machine-learning techniques [@JuantorenaAntisaccade]. Continuous gaze and pupil signals may subsequently be classified into eye-movement events. Commercial systems such as EyeLink provide vendor-reported event classifications through a built-in parser [@srresearch2021eyelink], whereas open algorithms include the methods of Engbert and Mergenthaler [@engbert2006microsaccades] and REMoDNaV [@dar2021remodnav].

The Brain Imaging Data Structure (BIDS) and the Eye-Tracking-BIDS extension now specify how sample-level gaze position, pupil recordings, and their metadata should be represented [@gorgolewski2016bids; @szinte2026eyetrackingbids]. However, converting heterogeneous vendor recordings into this representation and continuing from standardized samples to reproducible preprocessing, event detection, and analysis still requires substantial format-specific orchestration. Differences among experimental paradigms, devices, calibration procedures, and event-detection algorithms make it difficult to move from raw recordings to comparable results. Pyxations addresses this implementation gap while retaining the original vendor files for provenance.

In recent years, several open-source toolboxes have been developed to support eye-tracking data processing, including PyMovements [@krakowczyk2023pymovements], PyTrack [@ghose2020pytrack], SPEED [@lozzi2025speed], and PupEyes [@zhang2026pupeyes]. Each offers valuable functionality within the growing ecosystem of open tools for gaze analysis. Below we summarize their key features and how Pyxations extends or complements them.

PyMovements provides a modular Python interface for loading datasets, preprocessing gaze data, detecting oculomotor events, computing data-quality measures, and rendering plots. Its dataset definitions and configurable paths support both public and local datasets. Pyxations has a different emphasis: it converts supported vendor exports into paired raw and derivative BIDS datasets and carries that organization through preprocessing and hierarchical analysis.

PyTrack is an end-to-end toolkit featuring fixation, saccade, and microsaccade extraction, estimation of multiple metrics (including pupil and blinks), visualizations, and statistical analysis. Its graphical interface makes these workflows accessible to non-programmers. Pyxations complements that approach with explicit multi-vendor conversion, BIDS-organized raw and derivative datasets, and machine-readable preprocessing provenance.

SPEED (LabSoC Standardized Processing and Extraction of Eye-tracking Data) also focuses on lowering the entry barrier through a graphical feature-extraction workflow for Pupil Labs Neon data. Pyxations instead targets scriptable, multi-vendor conversion and analysis within a BIDS-organized workflow.

PupEyes provides a dedicated pupil-preprocessing and quality-control workflow, including deblinking, artifact rejection, smoothing, interpolation, baseline correction, and interactive diagnostic visualizations [@zhang2026pupeyes]. Pyxations does not duplicate this specialized pupillometry pipeline. Instead, it preserves pupil measurements and blink events when they are available, exposes them through `pupil_samples()` and `blinks()` at each level of its experiment hierarchy, and provides standardized tabular data that can be converted for use by downstream analysis packages.

Therefore, Pyxations was designed as a reproducible and extensible framework that unifies these complementary strengths while addressing their limitations. It is a Python-based toolbox designed to unify the organization, parsing, and analysis of eye-movement data. It supports multiple common recording formats and detection algorithms, and offers integrated tools for preprocessing, visualization, and downstream analysis. Its conversion layer writes per-eye BIDS physiological recordings and retains vendor originals under `sourcedata`; its processing layer writes processed samples and eye-movement annotations to a linked dataset following the general BIDS Derivatives conventions, facilitating reproducibility and comparison across tasks, devices, and studies.

# Research Impact Statement

Pyxations has already been deployed to standardize workflows in active research environments. It was utilized to process and analyze webcam-based eye-tracking data for a study on validation of an online antisaccade paradigm [@JuantorenaAntisaccade]. Moreover, source code was also utilized for EyeLink data in a study on hybrid search strategies, which combined Bayesian and neural network models [@ruarte2025integrating].

The library's versatility has been further evaluated through the harmonization of diverse datasets. It has been used to process data from GazePoint 3 trackers, in-house web-based eye-trackers, and driving simulation experiments (courtesy of collaborators mentioned in the acknowledgements), demonstrating its capacity to handle real-world heterogeneity beyond standard laboratory tasks. Furthermore, we are currently collaborating with three additional research groups to integrate Pyxations into their data analysis pipelines, providing additional opportunities to evaluate its portability and reproducibility.

# Software Design

Pyxations is designed as a reproducible and extensible framework that unifies the complementary strengths of previous developments while addressing their limitations. Its core contributions can be summarized (Fig. 1) as follows:

**Standardized dataset organization.** Pyxations writes sample-level eye-tracking data and metadata according to BIDS [@gorgolewski2016bids], including separate physiological recordings and corresponding tracker-event files for each eye, while retaining original vendor files under `sourcedata`. Processed sample streams and detected fixations, saccades, blinks, and messages are stored as compressed tabular files with JSON sidecars in a linked dataset that follows the general BIDS Derivatives conventions. BIDS does not yet define a domain-specific derivative schema for detected eye movements, so Pyxations documents its additional columns and processing provenance in those sidecars rather than claiming a separately standardized eye-movement derivative format. At the time of this revision, the generated raw and derivative datasets validate without errors against BIDS 1.11.1 using the official BIDS Validator 3.0.1 in continuous integration. This facilitates transparent sharing, version control, and collaborative reuse.

**Cross-format parsing and harmonization.** It supports EyeLink .edf/.asc, Tobii text exports, GazePoint CSV exports, and webcam recordings produced through WebGazer through a unified conversion API, reducing friction when combining data from different acquisition systems. Associated behavioral CSV/TSV files and PsychoPy trial logs can be normalized into BIDS events tables; PsychoPy-local timestamps are retained without assuming synchronization to the eye-tracker clock. Pupil measurements and blink events are retained when they are present in the source recording rather than being discarded during harmonization.

**Calibration- and validation-aware preprocessing.** Pyxations retains calibration reports and parses EyeLink VALIDATION blocks, extracting average and maximum error and offset values. These values can be inspected and used to exclude missing or poorly calibrated trials.

**Flexible trial segmentation.** The framework accommodates multiple trialing paradigms: explicit start and end timestamps, event-based markers, or fixed-duration trials. All of them with overlap controls and regular-expression message matching. This flexibility enables consistent parsing of diverse experimental logics without manual preprocessing.

**Declarative, provenance-aware workflow.** Configured trial-segmentation operations and their parameters are logged in machine-readable JSON recipes and provenance sidecars alongside the detection algorithm. This makes the transformations used to create each derivative dataset explicit and repeatable.

**Scalability and performance.** Pyxations uses the Polars data engine for in-memory preprocessing and analysis transformations [@vink2022] and can process independent recordings in parallel. These choices are intended to support multi-subject, multi-session analyses without making an unbenchmarked performance claim against other dataframe libraries.

**Visualization, statistics, and inspection tools.** In addition to standard gaze plots, Pyxations includes optional animated gaze visualizations, hierarchical data analysis (experiment, subject, session, trial), calibration-error visualization across trials, and task-specific visualization utilities tailored to paradigms such as visual search. Fixations, saccades, blinks, gaze samples, and pupil-containing samples can be retrieved at every level of the hierarchy. Optional MultiMatch metrics [@dewhurst2012] are available at the trial level to compare similarity between scanpaths.

**Interoperability.** Pyxations exposes standardized Polars tables for gaze samples, fixations, saccades, blinks, and pupil measurements, allowing downstream conversion or use by other Python analysis tools. It complements specialized packages such as PyMovements [@krakowczyk2023pymovements], PyTrack [@ghose2020pytrack], and PupEyes [@zhang2026pupeyes].

# Figure 1

![Preprocessing and analysis workflow showing parsing, event detection, preprocessing, and analysis stages.](figure1.png)


In summary, Pyxations extends current open-source eye-tracking software by offering a scalable, provenance-aware, and calibration-informed framework that unifies parsing, preprocessing, and analysis within a standardized data structure. Its detector selection and experiment hierarchy are designed to support further input formats and task-specific analyses without changing the canonical BIDS storage layer.


# AI Usage Disclosure

We utilized generative AI tools (primarily ChatGPT-4o and Gemini 2.5 Flash) to a limited extent, specifically for generating individual methods based on established architectural decisions, creating unit tests, and drafting documentation strings. We did not employ autonomous AI agents or large-scale automated coding pipelines. Regarding the manuscript, these tools were used solely for linguistic refinement, such as typo spotting, grammar correction, and minor stylistic improvements. The core scientific content, figures, and bibliography were not generated by AI.

# Acknowledgements

This project was supported by CONICET and UBA. We thank Pablo Laciana for his contribution to the code and Damián Care, Fermín Travi and Bruno Bianchi for their expert insights and constructive discussions during the preparation of this work. We thank Stephanie Muller and Enzo Tagliazucchi for sharing the GazePoint 3 data. Finally, We express our gratitude to Matias Ison for sharing the Driving data. 

# References
