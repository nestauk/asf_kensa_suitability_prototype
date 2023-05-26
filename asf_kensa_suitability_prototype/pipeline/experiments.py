# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: asf_kensa_suitability_prototype
#     language: python
#     name: asf_kensa_suitability_prototype
# ---

# %%
import os

os.chdir("../..")

# %%
from dask import dataframe as dd
import dask_geopandas
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import nearest_points

# %%
uprn_path = "inputs/data/uprn/osopenuprn_202304.csv"

# %%
uprn_df = dd.read_csv(uprn_path)
uprn_df["geometry"] = dask_geopandas.points_from_xy(
    uprn_df, "X_COORDINATE", "Y_COORDINATE"
)
uprn_gdf = dask_geopandas.from_dask_dataframe(uprn_df, geometry="geometry")

# %%
uprn = uprn_gdf.compute()

# %%
usrn_path = "inputs/data/usrn/osopenusrn_202305.gpkg"

# %% [markdown]
# the following doesn't work - IllegalArgumentException

# %%
# usrn_df = dask_geopandas.read_file(usrn_path, npartitions=4)
# usrn_df_computed = usrn_df.compute()

# %% [markdown]
# Restrict the geometries to a bounded area, both to make computation manageable and due to some geometries causing errors:

# %%
frame = (628160, 290000, 660000, 310000)
# xmin value chosen to avoid IllegalArgumentException - requires resolution

# %%
usrn = gpd.read_file(usrn_path, bbox=frame)
# only filters to geometries that intersect the bbox - clip with the frame to tidy up
bbox = box(*frame)
usrn = gpd.clip(usrn, mask=frame)

# %% [markdown]
# Plot of this part of the street network:

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn.plot(ax=ax)


# %%
uprn = gpd.clip(uprn, mask=frame)

# %% [markdown]
# Plot with streets and UPRNs:

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn.plot(ax=ax)
uprn.plot(ax=ax, markersize=2, color="orange")


# %% [markdown]
# ### Joining each UPRN to its nearest street

# %%
uprn = uprn.set_crs("EPSG:27700")

# %%
joined = gpd.sjoin_nearest(uprn, usrn, how="left", distance_col="distance")

# %%
# this automatically drops line geometries so re-add them
joined = joined.merge(
    usrn[["usrn", "geometry"]], on="usrn", suffixes=("_point", "_line")
)

# %%
joined = gpd.GeoDataFrame(joined)

# %% [markdown]
# note: many streets seem to appear in the USRN table twice, need to deduplicate

# %% [markdown]
# ### Ranking streets by number of UPRNs

# %%
counts = joined.groupby("usrn")["UPRN"].count().sort_values(ascending=False)
top_5_streets = list(counts.head().index)

# %%
usrn_not_top = usrn.loc[~usrn["usrn"].isin(top_5_streets)]
usrn_top = usrn.loc[usrn["usrn"].isin(top_5_streets)]


# %% [markdown]
# Top 5 streets with most UPRNs linked to them are marked in red on the plot:

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn_not_top.plot(ax=ax)
usrn_top.plot(ax=ax, color="red")
uprn.plot(ax=ax, markersize=2, color="orange")


# %% [markdown]
# Note: some of these streets are high-ranking because they are long. Need to account for density of UPRNs.

# %% [markdown]
# ### Ranking streets by density of UPRNs (number per unit length)

# %%
# add street lengths
joined["length"] = joined["geometry_line"].length

# %%
joined["UPRN_count"] = joined.groupby("usrn")["UPRN"].transform("count")
joined["UPRN_density"] = joined["UPRN_count"] / joined["length"]

# %%
# note: some streets with zero length, filter to avoid trivial cases
joined = joined.loc[joined["length"] > 0]

# %%
top_5_density_streets = list(
    joined.groupby("usrn")
    .head(1)
    .sort_values("UPRN_density", ascending=False)
    .head()["usrn"]
)

# %%
usrn_not_top_density = usrn.loc[~usrn["usrn"].isin(top_5_density_streets)]
usrn_top_density = usrn.loc[usrn["usrn"].isin(top_5_density_streets)]

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn_not_top_density.plot(ax=ax)
usrn_top_density.plot(ax=ax, color="red")
uprn.plot(ax=ax, markersize=2, color="orange")

# %% [markdown]
# Note: generally really tiny streets (barely visible on plot - two are in the lower right corner). Need to do something more involved, e.g. kernel density estimation.

# %% [markdown]
# ### Snapping UPRNs to nearest points on street network

# %%
joined["nearest_point"] = joined.apply(
    lambda x: nearest_points(x["geometry_point"], x["geometry_line"])[1], axis=1
)

# %%
joined.head()

# %% [markdown]
# Plot of UPRNs snapped to the street network (in red) just to check this is working:

# %%
f, ax = plt.subplots(figsize=(15, 15))
joined.set_geometry("geometry_point").plot(
    ax=ax, markersize=2, color="orange", zorder=1
)
joined.set_geometry("geometry_line").plot(ax=ax, zorder=2)
joined.set_geometry("nearest_point").plot(ax=ax, markersize=1, color="red", zorder=3)

# %%
# some points have a z-coordinate as the original linestrings have a z-coordinate
joined["nearest_point"].has_z.sum()

# %% [markdown]
# Note in the map above that there are some areas with a large number of UPRNs without any streets nearby. Potential risk here that streets surrounding these areas "absorb" lots of UPRNs that they are not actually close to (i.e. too far for Kensa to connect from from their nearest street). We can filter by distance to nearest street to avoid this problem. Also of course we need to filter UPRNs to just domestic properties and deal with things like blocks of flats

# %% [markdown]
# Next: build in linearity?

# %% [markdown]
#
