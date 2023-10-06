# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: asf_kensa_suitability_prototype
#     language: python
#     name: asf_kensa_suitability_prototype
# ---

# %%
import pandas
import geopandas
import numpy
from pyogrio import read_dataframe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.express as px
import boto3

# %%
# lsoa polygons from https://geoportal.statistics.gov.uk/maps/766da1380a3544c5a7ca9131dfd4acb6
lsoa_path = "../../inputs/data/lsoa/LSOA_Dec_2021_Boundaries_Generalised_Clipped_EW_BGC_V2_-6143458911719931546.gpkg"
# lsoa_path = "../../inputs/data/lsoa/LSOA_2021_EW_BSC_1975544170032131431.gpkg"
# datazone polygons from
datazone_path = "/vsizip/../../inputs/data/lsoa/SG_DataZoneBdry_2011.zip"

# Processed street density data path
street_data = "../../inputs/data/lsoa/uprn_street_density_lsoa_2021.csv"

# %%
# Load data
data = pandas.read_csv(street_data)
# geometries
# read LSOA data
lsoa = read_dataframe(lsoa_path)
datazone = read_dataframe(datazone_path)

# %%
# Combine lsoas and datazones
lsoa = pandas.concat(
    [
        lsoa[["LSOA21CD", "LSOA21NM", "geometry"]],
        datazone[["DataZone", "Name", "geometry"]].rename(
            columns={"DataZone": "LSOA21CD", "Name": "LSOA21NM"}
        ),
    ],
    ignore_index=True,
)
del datazone

# %%
# NB W01002024 (Cardiff 048F) is na - likely mixed use commercial/high density residential removed by uprn filtering step.
# likely not an issue for us here as mixed use urban probably not a target for share dground arrays.
lsoa = lsoa.merge(data, how="left", on="LSOA21NM").fillna(0)

# %%
lsoa["average_street_density"].describe()

# %%
density_percentiles = lsoa["average_street_density"].quantile(numpy.arange(0, 1, 0.01))

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(density_percentiles)
ax.set_xlabel("Density Percentile")
ax.set_ylabel("UPRNs per km")
ax.set_xticks(numpy.arange(0, 1.1, 0.1))
ax.grid()

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.plot(numpy.arange(0.01, 1, 0.01), numpy.log(density_percentiles.to_list()[1:]))
ax.set_xlabel("Density Percentile")
ax.set_ylabel("UPRNs per km")
ax.set_xticks(numpy.arange(0, 1.1, 0.1))
ax.grid()

# %%
plt.hist(
    numpy.log(
        lsoa.loc[lambda df: df["average_street_density"] > 0, "average_street_density"]
    ),
    bins=20,
)

# %%
lsoa["log_average_street_density"] = lsoa["average_street_density"].apply(
    lambda x: numpy.log(x) if x > 0 else 0
)

# %%
# lsoa.drop(columns='geometry').to_csv("../../outputs/tables/gb_lsoas_2021_dz_2011_uprn_usrn_density.csv", index=False)

# %%
f, ax = plt.subplots(figsize=(6, 9))

norm = LogNorm(
    vmin=lsoa.loc[
        lambda df: df["average_street_density"] > 0, "average_street_density"
    ].min(),
    vmax=lsoa["average_street_density"].max(),
)

lsoa.plot(
    column="average_street_density", edgecolor="none", norm=norm, cmap="coolwarm", ax=ax
)

cax = f.add_axes([0.7, 0.5, 0.03, 0.22])
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
# fake up the array of the scalar mappable.
sm._A = []
f.colorbar(sm, cax=cax)

# %%
lsoa_wgs84 = lsoa.to_crs(4326)

# %%
lsoa_wgs84.head()

# %%
lsoa_wgs84.to_file(
    "../../outputs/vectors/gb_lsoas_2021_dz_2011_uprn_usrn_density_wgs84.gpkg",
    driver="GPKG",
)

# %%
# Save file locally
lsoa_wgs84.to_parquet(
    "../../outputs/vectors/gb_lsoas_2021_dz_2011_uprn_usrn_density_qgs84.parquet"
)

# %%
lsoa_wgs84 = geopandas.read_parquet(
    "../../outputs/vectors/gb_lsoas_2021_dz_2011_uprn_usrn_density_qgs84.parquet"
)

# %%
# save file to public s3 bucket
key = "asf-kensa-prototype-lsoa.parquet"

s3 = boto3.client("s3")

s3.upload_file(
    Bucket="nesta-test",
    Filename="../../outputs/vectors/gb_lsoas_2021_dz_2011_uprn_usrn_density_qgs84.parquet",
    Key=key,
)

# %%
s3_url = "http://nesta-test.s3.amazonaws.com/asf-kensa-prototype-lsoa.parquet"

# %%
import fsspec

with fsspec.open(s3_url) as file:
    test = geopandas.read_parquet(
        file, columns=["log_average_street_density", "geometry"]
    )

# %%
test.head()

# %%
fig = px.choropleth_mapbox(
    lsoa_wgs84,
    geojson=lsoa_wgs84.geometry,
    locations=lsoa_wgs84.index,
    color="log_average_street_density",
    center={"lat": 54.3628, "lon": -3.4432},
    mapbox_style="open-street-map",
    zoom=12,
)
fig.show()
