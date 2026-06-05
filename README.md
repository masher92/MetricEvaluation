# MetricEvaluation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
[![DOI](https://zenodo.org/badge/935621052.svg)](https://doi.org/10.5281/zenodo.20554665)  

This repository contains code supporting the paper:
> **Critical assessment of metrics and methods used to quantify temporal loading of rainfall events**
> Molly Asher, Mark A. Trigg, Cathryn E. Birch, Rasmus L.T. Henriksen, Steven J. Böing, Jonas W. Pedersen
> *Hydrology and Earth System Sciences*, 2025
> 

### Background and overview
Rainfall event temporal loading — the distribution of rainfall intensity over time at a fixed location — significantly influences hydrological and geomorphological responses including urban flooding, runoff generation, and soil erosion. Despite this, a wide variety of metrics have been used to quantify temporal loading across the literature, with little systematic evaluation of how they relate to one another or how sensitive they are to data processing choices.

This codebase implements and evaluates 48 metrics identified through a structured literature review, plus 5 additional author-defined metrics, across five conceptual aspects of temporal loading. Code is provided for:

- Extracting rainfall events from rain gauge time series
- Computing 53 temporal loading metrics across those events
- Evaluating metric sensitivity to temporal resolution and dimensionless mass curve (DMC) transformation
- Clustering metrics to identify redundancy and complementarity
  
The codebase was applied to 233,128 rainfall events extracted from the Danish rain gauge network (1979–2025), recorded at 1-minute resolution and aggregated to 5-minute resolution for analysis.

---
Repository structure
```
MetricEvaluation/
│
├── 1. ExtractEvents/          # Scripts for extracting rainfall events from gauge data
│   └── ClassFunctions.py
│   └── FindEvents.py
│
├── 2. ProcessData/        # Implementation of all 53 temporal loading metrics
│   └── PreProcessEvents.ipynb
│   └── Scale_Transform.ipynb
│
├── 3. ClusterAnalysis/      # Agglomerative hierarchical clustering of metrics:
│   ├── AnalyseResults_MetricCrossover.ipynb   
│
├── 4. CompareNormalisation/         
│   └── MetricHistogram-DMCs.ipynb
│   └── MetricScatter-DMCs.ipynb
│   └── MetricHistogram-DblNorm.ipynb
│   └── MetricHistogram-DblNorm.ipynb
│
├── 5. CompareResolutions/                  
│   └── ByMetricHistogram.ipynb
│   └── ByMetricScatter.ipynb
│
├── 6. PlotsForPaper/                  # Scripts to reproduce paper figures
│   └── ...
│
└── requirements.txt          # Python dependencies
```
> **Note:** Please update this structure to reflect the actual directory layout of your repository.
---
Installation
Requirements
Python 3.8+
Dependencies listed in `requirements.txt`  

Setup  

```bash
git clone https://github.com/masher92/MetricEvaluation.git
cd MetricEvaluation
pip install -r requirements.txt
```
---
Usage
1. Event extraction
Rainfall events are extracted from gauge time series using a minimum inter-event time (MIT) threshold of 11 hours. Events with less than 4 mm total rainfall are excluded.
```bash
# Example — update with actual script name and arguments
python EventExtraction/extract_events.py --input <path_to_gauge_data> --output <output_dir>
```
2. Metric calculation
All 53 temporal loading metrics are calculated for each event. Metrics are implemented based on formulations in the original publications; where ambiguity exists, interpretations are documented in the code.
```bash
# Example — update with actual script name and arguments
python MetricCalculation/calculate_metrics.py --events <path_to_events> --output <output_dir>
```
3. Sensitivity to temporal aggregation
Metric values computed at 5-minute resolution are compared against those at 10-, 30-, and 60-minute resolution using sMAPE (numerical sensitivity) and Spearman's rank correlation (ranking sensitivity).
```bash
# Example — update with actual script name and arguments
python SensitivityAnalysis/TemporalAggregation/run_aggregation_sensitivity.py
```
4. Sensitivity to DMC transformation
Metric values are compared between raw 5-minute events and their double-normalised, 10-step DMC representations.
```bash
# Example — update with actual script name and arguments
python SensitivityAnalysis/DMCTransformation/run_dmc_sensitivity.py
```
5. Cluster analysis
Agglomerative hierarchical clustering based on pairwise Spearman rank correlation is used to identify groups of metrics describing similar properties.
```bash
# Example — update with actual script name and arguments
python ClusterAnalysis/run_clustering.py
```
---
### Data

Danish rain gauge data

This study uses rainfall data from the Danish Meteorological Institute (DMI) rain gauge network. DMI gauge observations are publicly available via the DMI Open Data API:
🔗 https://www.dmi.dk/frie-data (under "Meteorological Observations")
SVK (Water Pollution Committee of The Society of Danish Engineers) gauge data used in this study is not publicly available due to access restrictions. Access can be requested directly from SVK.
Input data format
> Please describe the expected input file format here (e.g. CSV with columns: timestamp, rainfall_mm; temporal resolution; file naming convention).

---

### Metrics implemented
The table below lists all 53 metrics implemented in this codebase, grouped by the five conceptual aspects of temporal loading identified in the paper.
Mass timing
`3rd/4th/5th with most`, `3rd with D50`, `3rd with CoG`, `Centre of gravity (CoG)`, `D50`, `T25`, `T75`, `m3`, `m4`, `m5`, `Temporal skewness`, `Event loading index`, `Asymmetry of dependence`, `Frac. in Q1/Q2/Q3/Q4`
Peak timing
`3rd/4th/5th with peak`, `3rd ppr`, `Time to peak`, `Peak position ratio`, `Skewp`, `m1`
Magnitude concentration
`Max intensity`, `I30`, `PCI`, `Classical skewness`, `Classical kurtosis`, `Classical std`, `CV`, `Mean intensity in HIZ`, `% time in LIZ/HIZ`, `% rain in HIZ`, `Gini coefficient`, `Lorenz asymmetry coefficient`, `NRMSEp`, `Peak-mean ratio`, `Relative amplitude`, `m2`, `Event dry ratio`
Temporal concentration
`TCI`, `Temporal kurtosis`, `Temporal standard deviation`
Intermittency
`Wet-dry transition rate`
Diagnostic (not temporal loading metrics)
`Mean intensity`, `Max intensity`
Full mathematical definitions for each metric are provided in the paper (Tables 1 and 2) and in the code documentation.
---
Reproducing paper results
All figures and analyses in the paper can be reproduced using the scripts in `Figures/`. Results are based on the full 233,128-event dataset; a smaller example dataset is provided for testing purposes.
> Please add any additional notes here about figure reproduction, e.g. expected runtime, memory requirements.
---
Citation
If you use this code, please cite:
```bibtex
@article{asher2025temporal,
  title   = {Critical assessment of metrics and methods used to quantify
             temporal loading of rainfall events},
  author  = {Asher, Molly and Trigg, Mark A. and Birch, Cathryn E. and
             Henriksen, Rasmus L.T. and B{\"o}ing, Steven J. and Pedersen, Jonas W.},
  journal = {Hydrology and Earth System Sciences},
  year    = {2025},
  doi     = {ADD DOI HERE}
}
```
---
Licence
This code is released under the MIT Licence. See `LICENSE` for details.
---
Contact
Molly Asher — kv25483@bristol.ac.uk
Issues and pull requests are welcome.
