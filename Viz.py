#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align: center">AIPI 510 Project 1</h1>
# 
# <h2>Happiness Index of Countries Over Time</h2>
# <b>Author(s):</b> Shreya Mendi, Yash Bhargava
# <br>
# <b>Date:</b> 9/30/2025

# <h2>Data Source</h2>
# <p>Put source of data here</p>

# <h3>About Dataset</h3>
# <p>The dataset contains yearly happiness report by country and contains 13 columns including:
# <ul>
#     <li>Year</li>
#     <li>Rank</li>
#     <li>Country name</li>
#     <li>Ladder Score</li>
#     <li>upperwhisker</li>
#     <li>lowerwhisker</li>
#     <li>Explained by: Log GDP per capita</li>
#     <li>Explained by: Social support</li>
#     <li>Explained by: Health life expectancy</li>
#     <li>Explained by: Freedom to make life choices</li>
#     <li>Explained by: Generosity</li>
#     <li>Explained by: Perceptions of corruption</li>
#     <li>Dystopia + residual</li>
# </ul>
# </p>

# <h3>Methodology</h3>
# <p>Several steps were taken including:<br>Data preprocessing to clean and prepare the data<br>Exploratory data analysis (EDA)<br>Feature engineering and transformation</p>

# <h4>Import Libraries</h4>

# In[159]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


# <h4>Load Data</h4>

# In[160]:


df = pd.read_excel("WHR25_Data_Figure_2.1.xlsx")
df.info()


# In[161]:


df.describe()


# In[162]:


df.head()


# <h4>Data Preprocessing</h4>
# <h5>Mapping countries to create a region column</h5>

# In[163]:


region_map = {
    # Carribean
    "Cuba": "Carribean",
    # Europe
    "Belarus": "Europe",
    "Finland": "Europe",
    "Denmark": "Europe",
    "Iceland": "Europe",
    "Sweden": "Europe",
    "Netherlands": "Europe",
    "Republic of Moldova": "Europe",
    "Malta": "Europe",
    "Norway": "Europe",
    "Luxembourg": "Europe",
    "Switzerland": "Europe",
    "Macedonia": "Europe",
    "Belgium": "Europe",
    "Ireland": "Europe",
    "Lithuania": "Europe",
    "Austria": "Europe",
    "Slovenia": "Europe",
    "Czechia": "Europe",
    "Germany": "Europe",
    "United Kingdom": "Europe",
    "Poland": "Europe",
    "France": "Europe",
    "Romania": "Europe",
    "Spain": "Europe",
    "Estonia": "Europe",
    "Italy": "Europe",
    "Latvia": "Europe",
    "Slovakia": "Europe",
    "Portugal": "Europe",
    "Hungary": "Europe",
    "Montenegro": "Europe",
    "Croatia": "Europe",
    "Greece": "Europe",
    "Bulgaria": "Europe",
    "North Macedonia": "Europe",
    "Albania": "Europe",
    "Serbia": "Europe",
    "Bosnia and Herzegovina": "Europe",
    "Ukraine": "Europe",
    "Kosovo": "Europe",

    # North America
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Belize": "North America",

    # Central & South America
    "Costa Rica": "Latin America",
    "Brazil": "Latin America",
    "El Salvador": "Latin America",
    "Puerto Rico": "Latin America",
    "Guyana": "Latin America",
    "Panama": "Latin America",
    "Argentina": "Latin America",
    "Guatemala": "Latin America",
    "Suriname": "Latin America",
    "Chile": "Latin America",
    "Nicaragua": "Latin America",
    "Paraguay": "Latin America",
    "Uruguay": "Latin America",
    "Ecuador": "Latin America",
    "Honduras": "Latin America",
    "Colombia": "Latin America",
    "Peru": "Latin America",
    "Bolivia": "Latin America",
    "Dominican Republic": "Latin America",
    "Venezuela": "Latin America",
    "Jamaica": "Latin America",
    "Trinidad and Tobago": "Latin America",
    "Haiti": "Latin America",  # if present

    # Asia-Pacific
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Israel": "Middle East",
    "United Arab Emirates": "Middle East",
    "Saudi Arabia": "Middle East",
    "Kuwait": "Middle East",
    "Bahrain": "Middle East",
    "Oman": "Middle East",
    "Qatar": "Middle East",  # if present
    "Turkey": "Middle East",
    "Türkiye": "Middle East",
    "State of Palestine": "Middle East",
    "Iran": "Middle East",
    "Iraq": "Middle East",
    "Jordan": "Middle East",
    "Lebanon": "Middle East",

    # Asia
    "Afghanistan": "Asia",
    "Bhutan": "Asia",
    "China": "Asia",
    "India": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Turkmenistan": "Asia",
    "Nepal": "Asia",
    "Viet Nam": "Asia",
    "Sri Lanka": "Asia",
    "Myanmar": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "Cambodia": "Asia",
    "Lao PDR": "Asia",
    "Indonesia": "Asia",
    "Malaysia": "Asia",
    "Singapore": "Asia",
    "Philippines": "Asia",
    "Japan": "Asia",
    "Republic of Korea": "Asia",
    "Taiwan Province of China": "Asia",
    "Mongolia": "Asia",
    "Kazakhstan": "Asia",
    "Uzbekistan": "Asia",
    "Kyrgyzstan": "Asia",
    "Tajikistan": "Asia",
    "Armenia": "Asia",
    "Azerbaijan": "Asia",
    "Georgia": "Asia",
    "Hong Kong SAR of China": "Asia",
    "Maldives": "Asia",  # if present

    # Africa
    "Angola": "Africa",
    "Burundi": "Africa",
    "Central African Republic": "Africa",
    "Djibouti": "Africa",
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Ghana": "Africa",
    "Gambia": "Africa",
    "South Sudan": "Africa",
    "Togo": "Africa",
    "Rwanda": "Africa",
    "Kenya": "Africa",
    "Somaliland Region": "Africa",
    "Uganda": "Africa",
    "Tanzania": "Africa",
    "Ethiopia": "Africa",
    "Swaziland": "Africa",
    "Zimbabwe": "Africa",
    "Zambia": "Africa",
    "Malawi": "Africa",
    "Mozambique": "Africa",
    "Cameroon": "Africa",
    "Senegal": "Africa",
    "Namibia": "Africa",
    "Côte d’Ivoire": "Africa",
    "Guinea": "Africa",
    "Chad": "Africa",
    "Mali": "Africa",
    "Somalia": "Africa",
    "Mauritania": "Africa",
    "Burkina Faso": "Africa",
    "Benin": "Africa",
    "Liberia": "Africa",
    "Sierra Leone": "Africa",
    "Madagascar": "Africa",
    "Gabon": "Africa",
    "Congo": "Africa",
    "DR Congo": "Africa",
    "Botswana": "Africa",
    "Niger": "Africa",
    "Eswatini": "Africa",
    "Lesotho": "Africa",
    "Comoros": "Africa",
    "Mauritius": "Africa",
    "Libya": "Africa",
    "Egypt": "Africa",
    "Algeria": "Africa",
    "Tunisia": "Africa",
    "Morocco": "Africa",
    "Sudan": "Africa",   # if present
    "Yemen": "Middle East/Africa",  # sometimes grouped
    "Cyprus": "Middle East",
    "Syria": "Middle East",
    "North Cyprus": "Middle East",
    "Russian Federation": "Europe/Asia"

}


# In[164]:


# Mapping countries to regions
df["Region"] = df["Country name"].map(region_map)
# Cleaning up column names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace(":", "")
)
# Changed ranking values by 1 as all data was shifted by 1
df.loc[df["year"] == 2020, "rank"] = df.loc[df["year"] == 2020, "rank"] - 1
# Checking null values per column
df.isnull().sum()


# <h5>Checking for duplicate data</h5>

# In[165]:


# Checking for duplicate data
df.duplicated(subset=["year", "country_name"]).sum()


# <h5>Checking for years that have missing data, this will help with data visualizations as well as any necessary data cleaning needed</h5>

# In[166]:


missing_values_by_year_summary = df.groupby('year').agg(lambda x: x.isna().sum())
missing_values_by_year_summary


# <h5>We can see a lot of data missing is between years 2011-2018, as a result we will just use data from 2019 and up and clean that</h5>

# In[167]:


new_df = df[df['year'].between(2019,2024)].copy()
# Sort data in ascending order for years for each country
new_df = new_df.sort_values(by=["country_name", 'year'])
new_df.isnull().sum()


# In[168]:


rows_with_missing = new_df[new_df.isnull().any(axis=1)]

rows_with_missing


# In[169]:


countries_missing = ['Bahrain', 'Oman', 'State of Palestine', 'Tajikistan']
for country in countries_missing:
    print(f'Number of entries for {country}: ', len(new_df[new_df['country_name'] == country]))


# In[170]:


# Bahrain data
bahrain = (new_df['country_name'] == 'Bahrain') & (new_df['year'].between(2019, 2022))
bahrain_averages = new_df.loc[bahrain].mean(numeric_only=True)

bahrain_2023 = (new_df['country_name'] == 'Bahrain') & (new_df['year'] == 2023)
new_df.loc[bahrain_2023] = new_df.loc[bahrain_2023].fillna(bahrain_averages)

# State of Palestine data
sp = (new_df['country_name'] == 'State of Palestine') & (new_df['year'].between(2019, 2021))
sp_averages = new_df.loc[sp].mean(numeric_only=True)

sp_2022_2024 = (new_df['country_name'] == 'State of Palestine') & (new_df['year'].between(2022,2024))
new_df.loc[sp_2022_2024] = new_df.loc[sp_2022_2024].fillna(sp_averages)

# Tajikstan data
tajikstan = (new_df['country_name'] == 'Tajikistan') & (new_df['year'].between(2019, 2022))
tajikstan_averages = new_df.loc[tajikstan].mean(numeric_only=True)

tajikstan_2023_2024 = (new_df['country_name'] == 'Tajikistan') & (new_df['year'].between(2023,2024))
new_df.loc[tajikstan_2023_2024] = new_df.loc[tajikstan_2023_2024].fillna(tajikstan_averages)

# Oman is a special case - Fill with regional averages
numeric_cols = new_df.select_dtypes('number').columns
new_df[numeric_cols] = new_df.groupby('region')[numeric_cols].transform(lambda x: x.fillna(x.mean()))

# Re-verify no missing data
new_df.isnull().sum()


# In[171]:


# Save cleaned data as a new file
new_df.to_csv("WHR25_Data_Figure_2.1_cleaned.xlsx")


# <h4>Exploratory Data Analysis</h4>

# In[172]:


years = sorted(new_df["year"].unique())
all_frames = []

for i in range(len(years) - 1):
    year_start, year_end = years[i], years[i+1]
    df_start = new_df[new_df["year"] == year_start].set_index("country_name")
    df_end = new_df[new_df["year"] == year_end].set_index("country_name")

    # Align by country (drop if missing in either year)
    common_countries = df_start.index.intersection(df_end.index)
    df_start, df_end = df_start.loc[common_countries], df_end.loc[common_countries]

    # Generate smooth intermediate frames (10 steps per year)
    steps = 10
    for j in range(steps):
        alpha = j / steps
        df_interp = df_start.copy()
        for col in ["ladder_score", "explained_by_log_gdp_per_capita", "explained_by_healthy_life_expectancy"]:
            df_interp[col] = (1-alpha)*df_start[col] + alpha*df_end[col]
        df_interp["year"] = year_start + alpha*(year_end - year_start)
        all_frames.append(df_interp.reset_index())

# Add the last year
all_frames.append(new_df[new_df["year"] == years[-1]])

# Concatenate all interpolated frames
df_smooth = pd.concat(all_frames, ignore_index=True)

# -----------------------------
# Plotly Animated Scatter
# -----------------------------
fig = px.scatter(
    df_smooth,
    x="explained_by_log_gdp_per_capita",
    y="explained_by_healthy_life_expectancy",
    size="ladder_score",
    color="region",
    hover_name="country_name",
    animation_frame=df_smooth["year"].round(1).astype(str),  # smooth fractional years
    animation_group="country_name",
    size_max=50,
    range_x=[new_df["explained_by_log_gdp_per_capita"].min()-0.1, new_df["explained_by_log_gdp_per_capita"].max()+0.1],
    range_y=[new_df["explained_by_healthy_life_expectancy"].min()-0.1, new_df["explained_by_healthy_life_expectancy"].max()+0.1],
    title="🌍 World Happiness Report — GDP vs Health"
)

# Make animation smooth & slow
fig.update_layout(
    height=800, width=1100,
    font=dict(size=16),
    title_font=dict(size=24),
    plot_bgcolor="white"
)
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300  # slower per frame
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300  # smooth ease

fig.show()


# ### Inference: GDP vs Health
# This visualization shows that while GDP and life expectancy often move together, they don’t fully explain happiness.  
# - Wealthy countries like the U.S. and Japan don’t always outperform on health.  
# - Smaller countries like Costa Rica achieve strong life expectancy despite lower GDP.  
# - Happiness rankings require looking beyond these two factors alone.
# 

# In[173]:


rankings = new_df[["year", "country_name", "rank", "ladder_score", "region"]].sort_values(["year", "rank"])

years = sorted(new_df["year"].unique())
all_frames = []

for i in range(len(years) - 1):
    year_start, year_end = years[i], years[i+1]
    df_start = new_df[new_df["year"] == year_start].set_index("country_name")
    df_end = new_df[new_df["year"] == year_end].set_index("country_name")

    # Align by country (drop if missing in either year)
    common_countries = df_start.index.intersection(df_end.index)
    df_start, df_end = df_start.loc[common_countries], df_end.loc[common_countries]

    # Generate smooth intermediate frames (10 steps per year)
    steps = 10
    for j in range(steps):
        alpha = j / steps
        df_interp = df_start.copy()
        for col in ["ladder_score", "rank"]:
            df_interp[col] = (1-alpha)*df_start[col] + alpha*df_end[col]
        df_interp["year"] = year_start + alpha*(year_end - year_start)
        all_frames.append(df_interp.reset_index())

# Add final year
all_frames.append(new_df[new_df["year"] == years[-1]])

# Concatenate interpolated frames
df_smooth = pd.concat(all_frames, ignore_index=True)

# Round year labels for animation
df_smooth["YearLabel"] = df_smooth["year"].round(1).astype(str)

# -----------------------------
# Plotly Bar Chart Race
# -----------------------------
fig = px.bar(
    df_smooth,
    x="ladder_score",
    y="country_name",
    orientation="h",
    color="region",
    animation_frame="YearLabel",
    animation_group="country_name",
    range_x=[0, df["ladder_score"].max() + 1],
    title="🌍 World Happiness Rankings- All Countries",
    height=1400, width=1100
)

# Make Rank 1 appear at top
fig.update_layout(
    yaxis={'categoryorder':'total ascending'},
    font=dict(size=12),
    title_font=dict(size=24),
    plot_bgcolor='white'
)

# Slow down animation
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 3000
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 3000

fig.show()


# ### Inference: Global Ranking Trends
# The bar chart race highlights dramatic shifts:  
# - Nordic countries (Finland, Denmark, Iceland) consistently stay at the top.  
# - War/conflict regions like Afghanistan and Lebanon remain at the bottom.  
# - Some countries (e.g., Eastern Europe, Latin America) move noticeably year-to-year, showing the effect of political and social changes.
# 

# In[174]:


region_trends = new_df.groupby(["year","region"])["ladder_score"].mean().reset_index()

fig = px.line(
    region_trends,
    x="year", y="ladder_score",
    color="region",
    markers=True,
    title="Average Happiness Score by Region"
)
fig.show()


# ### Inference: Regional Averages
# - On average,Oceania, Europe and North America remain high but stable.  
# - Latin America shows resilience: moderate GDP but above-average happiness.  
# - Africa and South Asia consistently remain lower, reflecting structural inequalities.  
# 
# This suggests cultural and institutional factors, not just wealth, drive happiness.
# 

# In[175]:


heatmap_data = df.pivot(index="country_name", columns="year", values="rank")

fig = px.imshow(
    heatmap_data,
    aspect="auto",
    color_continuous_scale="YlGnBu_r",
    title="Country Rank Heatmap (Lower Rank = Happier)",
    labels=dict(color="rank")
)
fig.update_yaxes(autorange="reversed")  # so Rank 1 is at the top
fig.show()


# ### Inference: Heatmap of Ranks
# - This heatmap makes the stability of Nordic countries clear.  
# - Finland has been at the very top every year.  
# - Countries in crisis (Afghanistan, Lebanon) stay consistently at the bottom.  
# - Middle-tier countries (India, China) show gradual but steady improvements.
# 

# In[176]:


corr = new_df[[
    "ladder_score",
    "explained_by_log_gdp_per_capita",
    "explained_by_social_support",
    "explained_by_healthy_life_expectancy",
    "explained_by_freedom_to_make_life_choices",
    "explained_by_generosity",
    "explained_by_perceptions_of_corruption"
]].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation of Happiness Score with Explanatory Factors"
)
fig.show()


# ### Inference: What Drives Happiness?
# The strongest correlations with happiness are:  
# - Social support  
# - Healthy life expectancy  
# - Freedom to make life choices  
# GDP matters, but corruption perception and generosity show weaker but still notable links.  
# This confirms the paradox: **money helps, but trust and community matter more**.
# 

# In[177]:


top10 = new_df[new_df['year']==2024].nlargest(10, 'ladder_score')

fig = px.bar(
    top10,
    x="ladder_score", y="country_name",
    orientation="h",
    color="region",
    title="Top 10 Happiest Countries (2024)"
)
fig.show()


# In[178]:


fig = px.line(
    new_df[new_df['country_name'].isin(top10['country_name'])],
    x="year", y="ladder_score", color="country_name",
    title="Ladder Score Trends Over Time (Top 10 Countries of 2024)",
    markers=True
)
fig.show()


# In[179]:


bottom10 = new_df[new_df['year']==2024].nsmallest(10, 'ladder_score')

fig = px.bar(
    bottom10,
    x="ladder_score", y="country_name",
    orientation="h",
    color="region",
    title="Top 10 Least Happiest Countries (2024)"
)
fig.show()


# In[180]:


fig = px.line(
    new_df[new_df['country_name'].isin(bottom10['country_name'])],
    x="year", y="ladder_score", color="country_name",
    title="Ladder Score Trends Over Time (Top 10 Least Happiest Countries of 2024)",
    markers=True
)
fig.show()


# ### Inference: Top vs. Bottom Countries
# - The Top 10 are dominated by Nordic countries and a few others with strong institutions.  
# - The Bottom 10 are conflict-driven or resource-poor countries.  
# - Clear pattern: stability, equality, and social trust separate the happiest from the unhappiest.
# 

# In[181]:


years = sorted(new_df["year"].unique())
all_frames = []

for i in range(len(years) - 1):
    year_start, year_end = years[i], years[i+1]
    df_start = new_df[new_df["year"] == year_start].set_index("country_name")
    df_end = new_df[new_df["year"] == year_end].set_index("country_name")

    # Align by country (drop if missing in either year)
    common_countries = df_start.index.intersection(df_end.index)
    df_start, df_end = df_start.loc[common_countries], df_end.loc[common_countries]

    # Generate smooth intermediate frames (10 steps per year)
    steps = 10
    for j in range(steps):
        alpha = j / steps
        df_interp = df_start.copy()
        for col in ["ladder_score", "explained_by_freedom_to_make_life_choices", "explained_by_social_support"]:
            df_interp[col] = (1-alpha)*df_start[col] + alpha*df_end[col]
        df_interp["year"] = year_start + alpha*(year_end - year_start)
        all_frames.append(df_interp.reset_index())

# Add the last year
all_frames.append(new_df[new_df["year"] == years[-1]])

# Concatenate all interpolated frames
df_smooth = pd.concat(all_frames, ignore_index=True)

# -----------------------------
# Plotly Animated Scatter
# -----------------------------
fig = px.scatter(
    df_smooth,
    x="explained_by_freedom_to_make_life_choices",
    y="explained_by_social_support",
    size="ladder_score",
    color="region",
    hover_name="country_name",
    animation_frame=df_smooth["year"].round(1).astype(str),  # smooth fractional years
    animation_group="country_name",
    size_max=50,
    range_x=[new_df["explained_by_freedom_to_make_life_choices"].min()-0.1, new_df["explained_by_freedom_to_make_life_choices"].max()+0.1],
    range_y=[new_df["explained_by_social_support"].min()-0.1, new_df["explained_by_social_support"].max()+0.1],
    title="🌍 World Happiness Report — Social support vs Freedom of choice"
)

# Make animation smooth & slow
fig.update_layout(
    height=800, width=1100,
    font=dict(size=16),
    title_font=dict(size=24),
    plot_bgcolor="white"
)
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300  # slower per frame
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300  # smooth ease

fig.show()


# ### Inference: Social Support vs. Freedom of Choice
# 
# This chart highlights how **social trust and freedom** explain happiness more than GDP:  
# 
# - Countries in the **top-right** (high social support, high freedom) have the **largest bubbles** (highest happiness).  
#   - Example: **Finland, Denmark, Iceland** — balanced across both dimensions.  
# 
# 
# - Some countries in **Latin America** (e.g., Costa Rica) show **above-average happiness** even with modest GDP, thanks to strong community ties and perceived freedom.  
# 
# **Key Insight:**  
# Happiness is highest when people feel supported and free to make choices — two factors often underestimated compared to economic growth.
# 

# # Cross-Country Comparisons
# 1. The Consistent Champions 

# In[182]:


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

countries = ["Finland", "Denmark", "Iceland", "Sweden", "Netherlands"]

# Short labels for readability
factors = {
    "explained_by_log_gdp_per_capita": "GDP",
    "explained_by_social_support": "Support",
    "explained_by_healthy_life_expectancy": "Health",
    "explained_by_freedom_to_make_life_choices": "Freedom",
    "explained_by_generosity": "Generosity",
    "explained_by_perceptions_of_corruption": "Corruption"
}

years = sorted(new_df["year"].unique())

# --- Helper to build radar traces for a given year ---
def get_traces_for_year(year):
    traces = []
    for country in countries:
        row = new_df[(new_df["year"] == year) & (new_df["country_name"] == country)]
        if not row.empty:
            values = row[list(factors.keys())].values.flatten().tolist()
            values.append(values[0])  # close loop
            labels = list(factors.values()) + [list(factors.values())[0]]
            traces.append(go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=country,
                line=dict(width=2),
                mode="lines+markers",
                marker=dict(size=6)
            ))
        else:
            traces.append(None)
    return traces

# --- Subplot grid (3 cols × 2 rows = 6 slots, use 5) ---
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"type": "polar"}]*3, [{"type": "polar"}]*3],
    subplot_titles=countries
)

positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]

# Add initial traces (first year)
initial_traces = get_traces_for_year(years[0])
for trace, pos in zip(initial_traces, positions):
    if trace:
        fig.add_trace(trace, row=pos[0], col=pos[1])

# --- Frames for animation ---
frames = []
for year in years:
    traces = get_traces_for_year(year)
    frame_data = [t for t in traces if t]
    frames.append(go.Frame(data=frame_data, name=str(year)))

fig.frames = frames

# --- Layout polish ---
fig.update_layout(
    title="Happiness Factors (2019–2024) — Consistently top Countries",
    template="plotly_white",
    showlegend=False,
    title_x=0.5,
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "▶ Play", "method": "animate",
             "args": [None, {"frame": {"duration": 1000, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 500}}]},
            {"label": "⏸ Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
        ]
    }],
    sliders=[{
        "steps": [
            {"args": [[str(year)], {"frame": {"duration": 600, "redraw": True}, "mode": "immediate"}],
             "label": str(year), "method": "animate"}
            for year in years
        ],
        "currentvalue": {"prefix": "Year: ", "font": {"size": 16}}
    }],

    # ✅ Apply same axis settings to all polar subplots
    polar=dict(radialaxis=dict(visible=True, range=[0,2], showticklabels=False, ticks="")),
    polar2=dict(radialaxis=dict(visible=True, range=[0,2], showticklabels=False, ticks="")),
    polar3=dict(radialaxis=dict(visible=True, range=[0,2], showticklabels=False, ticks="")),
    polar4=dict(radialaxis=dict(visible=True, range=[0,2], showticklabels=False, ticks="")),
    polar5=dict(radialaxis=dict(visible=True, range=[0,2], showticklabels=False, ticks=""))
)


fig.show()


# ### Inference: Consistent Champions
# Finland, Denmark, Iceland, Sweden, and the Netherlands consistently rank at the top.  
# Radar charts show why: they balance GDP with high social support, strong health systems, low corruption, and freedom of choice.  
# These countries provide a model of resilience and balance.
# 

# 2. the falling gaints
# 

# In[183]:


# Pick your countries of interest
giants = ["United States", "France", "Italy", "Japan"]

df_giants = new_df[new_df["country_name"].isin(giants)]

fig = px.scatter(
    df_giants,
    x="explained_by_log_gdp_per_capita",
    y="rank",
    animation_frame="year",
    animation_group="country_name",
    color="country_name",
    size="explained_by_social_support",  # optional: bubble size = social support
    hover_name="country_name",
    range_x=[0.8, 2],   # adjust depending on your dataset
    range_y=[100,-10],     # adjust depending on your dataset
    title="The Falling giants: GDP per Capita vs. Happiness (2019–2024)"
)

fig.show()


# In[184]:


countries = ["United States", "France", "Italy", "Japan"]

factors = {
    "explained_by_log_gdp_per_capita": "GDP",
    "explained_by_social_support": "Support",
    "explained_by_healthy_life_expectancy": "Health",
    "explained_by_freedom_to_make_life_choices": "Freedom",
    "explained_by_generosity": "Generosity",
    "explained_by_perceptions_of_corruption": "Corruption",
}

years = sorted(new_df["year"].unique())

def get_traces_for_year(year):
    traces = []
    for country in countries:
        row = new_df[(new_df["year"] == year) & (new_df["country_name"] == country)]
        if not row.empty:
            values = row[list(factors.keys())].values.flatten().tolist()
            values.append(values[0])  # close loop
            labels = list(factors.values()) + [list(factors.values())[0]]
            traces.append(go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=country,
                line=dict(width=2),
                mode="lines+markers",
                marker=dict(size=6)
            ))
    return traces

# Grid layout for 6 countries
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"type": "polar"}]*3, [{"type": "polar"}]*3],
    subplot_titles=countries,
    horizontal_spacing=0.15,
    vertical_spacing=0.25,
)

positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

# Initial year
initial_traces = get_traces_for_year(years[0])
for trace, pos in zip(initial_traces, positions):
    fig.add_trace(trace, row=pos[0], col=pos[1])

# Animation frames
frames = []
for year in years:
    traces = get_traces_for_year(year)
    frames.append(go.Frame(data=traces, name=str(year)))

fig.frames = frames

# Layout polish
fig.update_layout(
    title="All Happiness Factors (2019–2024) — Falling Giants ",
    template="plotly_white",
    showlegend=False,
    annotations = [
        dict(
            text=country,
            x=pos[1]/3 - 0.16,
            y=1.12 if pos[0] == 1 else -0.18,   # ⬅️ bottom row pushed further down
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="black")
        )
        for country, pos in zip(countries, positions)
    ],
    polar=dict(radialaxis=dict(visible=True, range=[0, 3], showticklabels=False, ticks="")),
    polar2=dict(radialaxis=dict(visible=True, range=[0, 3], showticklabels=False, ticks="")),
    polar3=dict(radialaxis=dict(visible=True, range=[0, 3], showticklabels=False, ticks="")),
    polar4=dict(radialaxis=dict(visible=True, range=[0, 3], showticklabels=False, ticks="")),
    polar5=dict(radialaxis=dict(visible=True, range=[0, 3], showticklabels=False, ticks="")),
    polar6=dict(radialaxis=dict(visible=True, range=[0, 3], showticklabels=False, ticks="")),
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "▶ Play", "method": "animate",
             "args": [None, {"frame": {"duration": 1000, "redraw": True}, 
                             "fromcurrent": True, "transition": {"duration": 500}}]},
            {"label": "⏸ Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
        ]
    }],
    sliders=[{
        "steps": [
            {"args": [[str(year)], {"frame": {"duration": 700, "redraw": True}, "mode": "immediate"}],
             "label": str(year), "method": "animate"}
            for year in years
        ],
        "currentvalue": {"prefix": "Year: ", "font": {"size": 16}}
    }]
)

fig.show()


# ### Inference: Falling Giants
# The U.S., France, Italy, and Japan have strong GDP but weaker scores in social support, corruption perception, and health.  
# Their happiness rankings slip compared to Nordic countries.  
# This demonstrates the central paradox: **economic power without trust and freedom does not sustain happiness.**
# 

# ## Limitations & Ethics
# - Happiness scores are self-reported and culturally biased.  
# - Rankings may oversimplify complex social realities.  
# - Comparing countries should be done carefully to avoid stereotypes.
# 
# ## Takeaways
# - GDP is only part of the story.  
# - Social trust, freedom, and health are stronger drivers of happiness.  
# - Wealthy nations can still decline in well-being if these weaken.  
# - Countries like Finland and Costa Rica prove that balance, not just money, drives happiness.  
# 
# **Final Message:** Happiness isn’t bought—it’s built through community, trust, and fairness.
# 
