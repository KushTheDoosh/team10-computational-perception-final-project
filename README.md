# Team 10 — Intelligent Surveillance System
## ComS 5750: Computational Perception | Iowa State University

## Authors
- Sai Ujwal V (saiujwal@iastate.edu)
- Kush Chaudhary (kush@iastate.edu)

## Install Dependencies
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn streamlit

## Run the Pipeline
python3 03_evaluation/team10_eval_runner.py \
  --video 03_evaluation/animal.mp4 \
  --output-dir output/

## Note on FPS
All FPS results reported in the final report (51.89 FPS) were measured 
on a MacBook Air (Apple M4, CPU + GPU). Google Colab will show lower FPS 
(4-8 FPS) due to shared CPU resources. The algorithm and results are identical.

## Reproduce Full Evaluation
See 03_evaluation/README.md for the complete ground-truth labeling 
and evaluation workflow used to generate all reported metrics.

## Repository Structure
- 00_reports/          — Final report writeup
- 01_notebooks/        — Development notebooks  
- 02_project_docs/     — Project proposal
- 03_evaluation/       — Evaluation runner, videos, and results
