# Sourcing-Happiness

# AIPI 510 Project 1 — Happiness Index of Countries Over Time

## 📌 Overview
This project analyzes the **World Happiness Report** data to explore how happiness scores evolve across countries and regions over time (2019–2024).  
We investigate the role of key explanatory factors such as GDP, social support, healthy life expectancy, freedom of choice, generosity, and perceptions of corruption.  
Visualizations include animated scatter plots, bar chart races, heatmaps, and radar charts to highlight country-level, regional, and global trends.  

Key research questions addressed:
- Which countries and regions consistently rank highest and lowest in happiness?
- How do GDP, health, and social support correlate with happiness?
- What explains resilience in some countries despite low income levels?
- Are wealthy nations always the happiest?

---

## 📊 Dataset

- **Source**: *World Happiness Report 2025* (official dataset for Figure 2.1).  
- **File Used**: `WHR25_Data_Figure_2.1.xlsx`  
- **Years Covered**: 2019–2024 (data before 2019 omitted due to missingness).  
- **Variables (13 columns)**:
  - `Year`
  - `Rank`
  - `Country name`
  - `Ladder Score`
  - `upperwhisker`
  - `lowerwhisker`
  - `Explained by: Log GDP per capita`
  - `Explained by: Social support`
  - `Explained by: Healthy life expectancy`
  - `Explained by: Freedom to make life choices`
  - `Explained by: Generosity`
  - `Explained by: Perceptions of corruption`
  - `Dystopia + residual`

### Citation
> Helliwell, J. F., Layard, R., Sachs, J., De Neve, J.-E., Aknin, L., & Wang, S. (2025).  
> *World Happiness Report 2025*. Sustainable Development Solutions Network.  
> [https://worldhappiness.report](https://worldhappiness.report)

---

## ⚙️ Reproduction Instructions

### 1. Clone or download this repository
```bash
git clone https://github.com/Shreya-Mendi/Sourcing-Happiness
cd Sourcing-Happiness
````

### 2. Install required Python packages

We recommend using Python ≥ 3.9 in a virtual environment.

```bash
pip install pandas numpy seaborn matplotlib plotly openpyxl kaleido
```

* `openpyxl` → required to read Excel files (`.xlsx`)
* `kaleido` → required to save Plotly figures as `.png`

### 3. Place dataset in the project folder

Download `WHR25_Data_Figure_2.1.xlsx` from the [World Happiness Report 2025 site](https://www.worldhappiness.report/data-sharing/) and put it in the root project directory.

### 4. Run the Jupyter Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook
```

Then run all cells in `Viz.ipynb`.

### 5. Outputs

The notebook generates:

* Cleaned dataset: `WHR25_Data_Figure_2.1_cleaned.xlsx`
* Visualizations:

  * Animated scatter plots (`.html` + `.png`)
  * Bar chart race of rankings
  * Heatmap of ranks by year
  * Correlation matrix of factors
  * Regional line charts
  * Radar charts for “Consistent Champions” and “Falling Giants”
* All figures and animations saved in the `exports/` folder.

---

## 📈 Key Insights

* Nordic countries (Finland, Denmark, Iceland) consistently dominate the top rankings.
* GDP contributes to happiness, but **social trust, health, and freedom** are stronger drivers.
* Countries like Costa Rica achieve high happiness with modest GDP, thanks to strong social bonds.
* Wealthy nations like the U.S. and Japan underperform compared to peers, showing that money alone is insufficient.
* Regions with conflict (e.g., Afghanistan, Lebanon) remain at the bottom throughout the analysis period.

---

## ⚠️ Limitations & Ethics

* Happiness scores are self-reported and subject to cultural bias.
* Rankings can oversimplify complex societal conditions.
* Comparisons across countries should be interpreted cautiously.
