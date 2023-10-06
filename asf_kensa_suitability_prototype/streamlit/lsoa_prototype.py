# %%
import streamlit
import fsspec
import geopandas
import plotly.express as px

# %%
streamlit.set_page_config(page_title="LSOA UPRN Density Prototype", layout="wide")

# %%
streamlit.title("Density of UPRNs on USRNs for 2021 LSOAs and 2011 DataZones")

# %%
# Data Url
s3_url = "http://nesta-test.s3.amazonaws.com/asf-kensa-prototype-lsoa.parquet"

# %%
# Load Data
with fsspec.open(s3_url) as file:
    lsoa_wgs84 = geopandas.read_parquet(file)

# %%
fig = px.choropleth_mapbox(
    lsoa_wgs84,
    geojson=lsoa_wgs84.geometry,
    locations=lsoa_wgs84.index,
    color="log_average_street_density",
    center={"lat": 54.3628, "lon": -3.4432},
    mapbox_style="open-street-map",
    zoom=12,
    width=1600,
    height=800,
)

# %%
streamlit.plotly_chart(fig)
