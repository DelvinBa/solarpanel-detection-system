## 📌 Project Background

The energy transition is not just a technical challenge but a societal one. The **NOWATT** project aims to involve residents, SMEs, and government entities in accelerating energy neutrality at the neighborhood level using **Artificial Intelligence (AI)**. One critical aspect is **predicting the energy labels** of homes in the Netherlands.

A major factor influencing a home's energy efficiency is **solar panel installation**. By **detecting solar panels on rooftops**, we can improve **energy label prediction models** and help housing associations meet sustainability requirements by **2030**. Organizations like **Nijhuis Bouw** are actively working on these improvements.

This project leverages **computer vision** to detect solar panels using satellite imagery, forming a crucial step in energy label classification.

---

## 🛠️ Project Scope & Goals

In this project, we focus on:
- 📡 **Satellite Image Processing** – Detect solar panels from aerial/satellite images.
- 🏠 **Data Integration** – Align detected solar panels with **open-source** data like **Kadaster**.
- 🤖 **Object Detection Algorithms** – Experiment with **YOLO, Faster R-CNN, SSD**, etc.
- 🔄 **Automated ML Pipeline** – Develop a **data pipeline** that automates preprocessing, model training, and evaluation.
- 🚀 **API Deployment** – Convert the trained model into an **API** that accepts images and returns solar panel predictions.

You will work with:
✅ **Raw & annotated satellite image data**  
✅ **Preprocessing scripts & model training code**  
✅ **Object detection models (YOLO, etc.)**  

---

## ⚙️ Setup Guide



### **1️⃣ Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### **2️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

### **🛑 Configuring Git for Jupyter Notebooks**
Enable Automatic Output Stripping for Notebooks to prevent unnecessary notebook output changes in Git commits:
```
pre-commit install
```


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         solar_panel_detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes solar_panel_detection a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

