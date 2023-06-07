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
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: asf_kensa_suitability_prototype
#     language: python
#     name: asf_kensa_suitability_prototype
# ---

# %% [markdown]
# ### Imports and setup

# %%
import os

os.chdir("../..")

# %%
from dask import dataframe as dd
import dask_geopandas
import geopandas as gpd
from pyogrio import read_dataframe
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import nearest_points
from shapely import unary_union, line_merge
import contextily as cx

# %%
uprn_path = "inputs/data/uprn/osopenuprn_202304.csv"
# Using an fsspec file uri allows you to keep the data zipped.
# uprn_path = "zip://osopenuprn_202304.csv::inputs/data/uprn/osopenuprn_202305_csv.zip"

# Using an fsspec
# lids_path = "zip://BLPU_UPRN_Street_USRN_11.csv::inputs/data/lids/lids-2023-05_csv_BLPU-UPRN-Street-USRN-11.zip"
lids_path = "inputs/data/lids/BLPU_UPRN_Street_USRN_11.csv"

# usrn_path = "inputs/data/usrn/osopenusrn_202305.gpkg"
# GDAL virtual filesystem for reading from zipped geopackage file.
usrn_path = "/vsizip/inputs/data/usrn/osopenusrn_202306_gpkg.zip/osopenusrn_202306.gpkg"

# %% [markdown]
# Import UPRNs:

# %%
# Load uprns from zip file
uprn_df = dd.read_csv(uprn_path)

# %%
# Convert dask dataframe to geodataframe - defaults to 31 partitions from uprn_df.
uprn_gdf = dask_geopandas.from_dask_dataframe(
    uprn_df,
    geometry=dask_geopandas.points_from_xy(uprn_df, "X_COORDINATE", "Y_COORDINATE"),
)

# %%
uprn_gdf.head()

# %% [markdown]
# Import linked identifiers:

# %%
# Load lids
lids_df = dd.read_csv(lids_path)

# %%
# Identifier 1 is uprn, identifier 2 is usrn.
lids_df.head()

# %% [markdown]
# Join linked identifiers to UPRN data and explore missing.

# %%
uprn_df_check = uprn_df.merge(
    lids_df[["IDENTIFIER_1", "IDENTIFIER_2"]],
    how="left",
    left_on="UPRN",
    right_on="IDENTIFIER_1",
)

# %%
# The fact that IDENTIFIER_2 gets flipped to a float suggests that we're introducing some missing data.
uprn_df_check.head()

# %%
# Get count of na values - 127,600
uprn_df_check["IDENTIFIER_2"].isna().sum().compute()

# %%
# 127,600/ 41,123,153 = 0.31% missing
uprn_df_check["IDENTIFIER_2"].__len__()

# %%
missing = uprn_df_check.loc[lambda df: df["IDENTIFIER_2"].isna()].compute()

# Missing ids appear to be widely distributed, so I'll ignore them for now.
f, ax = plt.subplots(figsize=(5, 7))
missing.plot(x="X_COORDINATE", y="Y_COORDINATE", kind="scatter", marker=".", s=2, ax=ax)
ax.set_aspect("equal")

# %% [markdown]
# Join linked identifiers to UPRNs and group to summarise counts of UPRN per USRN.

# %%
# rejoin using inner to drop unmatched uprns
uprn_df = uprn_df.merge(
    lids_df[["IDENTIFIER_1", "IDENTIFIER_2"]],
    how="inner",
    left_on="UPRN",
    right_on="IDENTIFIER_1",
)

# %%
# Get count of UPRNs by USRNs
uprn_count = (
    uprn_df.groupby("IDENTIFIER_2")["IDENTIFIER_1"]
    .count()
    .compute()
    .reset_index(name="count")
)

# %%
# on average streets have 30 uprns associated, but median is 13 indicating skew. Max is high.
uprn_count["count"].describe()

# %% [markdown]
# Import USRNs:

# %%
# We'll drop 32 bad usrn's for now, however these could be fixed.
# 32 of 1,712,890 usrns is 0.002%
import fiona
from shapely.geometry import shape

# Find bad geoms
bad_usrns = []
with fiona.open(usrn_path) as src:
    for feat in src:
        try:
            shape(feat.geometry)
        except:
            bad_usrns.append(feat.properties["usrn"])

print(len(bad_usrns))

# %%
# They appear to be multilinestrings that have component lines with bad segments (e.g. a line made of 1 point).
# These could be fixed by dropping the bad segments and passing the rest to MultiLineString.
from shapely.geometry import LineString

with fiona.open(usrn_path) as src:
    for feat in src:
        if feat.properties["usrn"] in bad_usrns:
            count_bad_segments = 0
            for idx, line in enumerate(feat.geometry.coordinates):
                try:
                    LineString(line)
                except:
                    count_bad_segments += 1
            print(
                f"usrn : {feat.properties['usrn']} of type {feat.geometry.type} has \
{len(feat.geometry.coordinates)} segments of which {count_bad_segments} is/are malformed."
            )

# %%
# Read usrn data without problematic line features, using pyogrio.
usrn_gdf = read_dataframe(
    usrn_path, where=f"usrn not in ({','.join(map(str, bad_usrns))})"
)

# %%
# convert to dask geopandas geodataframe (nb 'where' not currently implemented in dask geopandas)
usrn_gdf = dask_geopandas.from_geopandas(usrn_gdf, npartitions=10)

# %%
usrn_gdf.head()

# %%
# merge uprn_count with usrn_gdf:
usrn_gdf_check = usrn_gdf.merge(
    uprn_count, how="outer", left_on="usrn", right_on="IDENTIFIER_2"
)

# %%
usrn_gdf_check.head()

# %%
# 670 Missing usrns of 1712890 - 0.04%
usrn_gdf_check.usrn.isna().sum().compute()

# %%
# missing data count of uprns sums to 9,411 of 40,995,553 - 0.02%
usrn_gdf_check.loc[lambda df: df.usrn.isna(), "count"].sum().compute()

# %%
# no matching uprns for 337,624 usrns
usrn_gdf_check.IDENTIFIER_2.isna().sum().compute()

# %%
# Rejoin the data retaining only matching rows.
usrn_gdf = usrn_gdf.merge(
    uprn_count, how="inner", left_on="usrn", right_on="IDENTIFIER_2"
)

# %%
# 1,375,234 usrns with joined uprn data - but might want to check/deal with duplicates.
usrn_gdf.__len__()

# %% [markdown]
# This is now an authoritative uprn - usrn joined dataset based on the the published linked identifiers data. It is a good starting point for simple analysis.
#
# However, as it cannot capture where a uprn is along a usrn segment it is only really useful at the usrn resolution.
#
# It's be good to look at some basic summaries of this data - e.g. uprns per length of usrn.
#
# Also, aggregating the data - perhaps to OA or LSOA would be interesting.
#
# As would creating a line density surface. You could do this using `rasterio` and iterating over each line feature, rasterising, adjusting the burnt in values according to the count/density of the uprns on the line and the number of cells covered, and then summing all the surfaces (in practice this would be more like a reduce operation). Then you could look at the raw surface, or smooth/convolve.

# %%

# %%

# %% [markdown]
# Restrict the geometries to a bounded area, both to make computation manageable and due to some geometries causing errors (the area corresponds to a section of the east coast between Great Yarmouth and Lowestoft):

# %%
frame = (628160, 290000, 660000, 310000)
# xmin value chosen to avoid IllegalArgumentException - in the real thing this would need fixing

# %%
usrn = gpd.read_file(usrn_path, bbox=frame)
# only filters to geometries that intersect the bbox - clip with the frame to tidy up
bbox = box(*frame)
usrn = gpd.clip(usrn, mask=frame)

# some duplicates - drop duplicate geometries
usrn = usrn.drop_duplicates("geometry")

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn.plot(ax=ax, alpha=0.5)
cx.add_basemap(ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik)
ax.set_title("Streets in filtered USRN dataset (blue)")

# %%
# do the same to UPRNs
uprn = dask_geopandas.clip(uprn_gdf, mask=frame).compute()

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn.plot(ax=ax, alpha=0.5)
uprn.plot(ax=ax, markersize=2, color="orange", alpha=0.5)
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title(
    "Locations in filtered UPRN dataset (orange) and streets in filtered USRN dataset (blue)"
)

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
# ### Ranking streets by number of nearby UPRNs

# %%
counts = joined.groupby("usrn")["UPRN"].count().sort_values(ascending=False)
top_5_streets = list(counts.head().index)
usrn_top = usrn.loc[usrn["usrn"].isin(top_5_streets)]

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn_top.plot(ax=ax, color="red")
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title("Top 5 streets with highest numbers of nearby UPRNs")

# %% [markdown]
# Note: some of these streets are high-ranking because they are long. Need to account for density of UPRNs.
#
# Note also that some streets are a weird shape! The cluster of roads in the bottom left seems to correspond to one USRN (83407249), and some are not connected.

# %% [markdown]
# ### Ranking streets by density of UPRNs (number per unit length)

# %%
# add street lengths
joined["length"] = joined["geometry_line"].length

# %%
joined["uprn_count"] = joined.groupby("usrn")["UPRN"].transform("count")
joined["uprn_density"] = joined["uprn_count"] / joined["length"]

# %%
top_5_density_streets = list(
    joined.groupby("usrn")
    .head(1)
    .sort_values("uprn_density", ascending=False)
    .head()["usrn"]
)

# %%
usrn_top_density = usrn.loc[usrn["usrn"].isin(top_5_density_streets)]

# %%
f, ax = plt.subplots(figsize=(15, 15))
usrn_top_density.plot(ax=ax, color="red", linewidth=5)
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title("Top 5 streets with highest UPRN density")

# %% [markdown]
# Generally really tiny streets, barely visible on plot. Need to do something more involved, e.g. kernel density estimation.

# %% [markdown]
# ### Snapping UPRNs to nearest points on street network

# %%
joined["nearest_point"] = joined.apply(
    lambda x: nearest_points(x["geometry_point"], x["geometry_line"])[1], axis=1
)

# %% [markdown]
# Plot of UPRNs (orange) and their equivalent points snapped to the street network (in red) just to check this is working - we should see that all red points fall on streets:

# %%
f, ax = plt.subplots(figsize=(15, 15))
joined.set_geometry("geometry_point").plot(
    ax=ax, markersize=2, color="orange", zorder=1, alpha=0.5
)
joined.set_geometry("geometry_line").plot(ax=ax, zorder=2, alpha=0.5)
joined.set_geometry("nearest_point").plot(
    ax=ax, markersize=1, color="red", zorder=3, alpha=0.5
)
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title("Streets (blue), UPRNs (orange) and UPRNs snapped to streets (red)")

# %% [markdown]
# Note in the map above that there are some areas with a large number of UPRNs without any streets nearby. Potential risk here that streets surrounding these areas "absorb" lots of UPRNs that they are not actually close to (i.e. too far for Kensa to connect from their nearest street). We can filter by distance to nearest street to avoid this problem. Also we need to remember to filter UPRNs to just domestic properties, and only those that are suitable (i.e. not blocks of flats).

# %% [markdown]
# ### Identifying dense street segments

# %% [markdown]
# One potential method would involve applying kernel density estimation on the snapped UPRNs to get a continuous density measure on the street network. We could then identify high-density areas by filtering to street segments that have density greater than a particular threshold.
#
# An approximation to this is as follows:
# * For each snapped UPRN, identify the part of the street that is less than a given distance *d* from the snapped UPRN
# * For each street, get the union of these parts
# * Find the connected components of this union
# * Calculate the total length of each component and calculate the average distance between each component and the UPRNs that are nearest to it
# * 'Suitable' street segments are therefore components that are sufficiently long and have a sufficiently small average distance-to-UPRN
#
# This is a bit like applying KDE with a uniform kernel. If we used KDE then overlapping kernels would be additive and we would get a better picture of how dense street segments are. But we don't really need this if we have a 'minimum allowable' density in mind - we just need to set *d* according to this value, then the street segments that result from this process will be 'sufficiently dense'.
#
# The limitation is that we need to know this value in advance, whereas if we used KDE with a non-uniform kernel then we'd get a more general output that could be filtered according to different density values. It's also more strict than KDE - in this method, if two snapped UPRNs are more than twice the 'given distance' apart with nothing in between, then the resulting street segments will be disconnected, but if we used KDE then we could choose a kernel that provides more flexibility, potentially leading to segments being joined if there are enough points on either side.
#
# We can further simplify the first step by just intersecting the street with a buffer circle of radius *d* around the snapped point. This is roughly the same thing, under the assumption that streets are approximately linear. This assumption should be checked though given that we know from the above that some streets are non-linear.

# %%
# for each point, add a buffer circle (arbitrary radius of 50 for this demo)
joined["buffer"] = joined.set_geometry("nearest_point").buffer(50)

# %%
# intersect each buffer circle with its corresponding street geometry
# need to intersect only with the corresponding street to avoid "spillover" to another nearby street
joined["segment"] = joined["buffer"].intersection(joined["geometry_line"])

# %%
# for each street, get the union of its buffer intersections
# the line merge cleans up the multi line strings by joining anything that should be connected
intersections = joined.groupby("usrn").apply(
    lambda x: line_merge(unary_union(x["segment"]))
)

intersections = intersections.reset_index().rename(columns={0: "geometry"})

# %%
f, ax = plt.subplots(figsize=(15, 15))

intersections.set_geometry("geometry").plot(ax=ax, color="red")
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title("Street segments with 'sufficiently high' UPRN density")

# %%
# explode multi line strings into individual line strings (connected components)
# to check: are line strings actually connected components? is there a better way to do this?
exploded = intersections.set_geometry("geometry").explode(ignore_index=True)

# %%
# set CRS for mapping
exploded = exploded.set_crs("EPSG:27700")

# %%
# join UPRN closest points onto their nearest line string
# there is probably a more efficient way to do this given that the line strings are based on the UPRNs
segment_counts = (
    joined[["nearest_point", "distance"]]
    .set_geometry("nearest_point")
    .sjoin_nearest(exploded)
)

# %%
# define new variables
exploded["uprn_count"] = segment_counts.groupby("index_right")["nearest_point"].count()
exploded["total_distance"] = segment_counts.groupby("index_right")["distance"].sum()

exploded["uprn_count"] = exploded["uprn_count"].fillna(0).astype(int)
exploded["total_distance"] = exploded["total_distance"].fillna(0)

exploded["average_distance"] = exploded["total_distance"] / exploded["uprn_count"]

exploded["length"] = exploded["geometry"].length
exploded["density"] = exploded["uprn_count"] / exploded["length"]

# %%
f, ax = plt.subplots(figsize=(15, 15))

exploded.plot(
    column="length", ax=ax, cmap="OrRd", legend=True, legend_kwds={"shrink": 0.3}
)
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title(
    "Street segments with 'sufficiently high' UPRN density, coloured by length"
)

# %%
fig, ax = plt.subplots(figsize=(15, 15))

# arbitrary parameters for now
exploded.loc[
    (exploded["length"] > 100)
    & (exploded["density"] > 0.2)
    & (exploded["average_distance"] < 50)
].plot(ax=ax, color="red")
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)
ax.set_title(
    "Street segments with 'sufficient' length, UPRN density and average distance to UPRN"
)

# %% [markdown]
# Still a few issues to resolve - lots of the high-density segments are really small when looking at the full dataset, does this indicate problems with the underlying geometry? Possibly small segments that are detached from the rest? Disconnected line strings? Or points where streets change direction near a cluster of UPRNs, so they all get joined to a corner?

# %% [markdown]
# ## Key things to resolve

# %% [markdown]
# * Some streets are weird shapes - loops, forks, etc. This causes problems with the "buffer circle" approximation - would be better to "crawl" outward from the snapped point to find all points on the street that are within a certain distance along the street (not as the crow flies)
#
# * Generally check street geometries to make sure line segments are connected properly
#
# * Need to check that deduplicating streets by geometry is robust - why are streets duplicated in the first place?
#
# * Filter to domestic UPRNs of suitable built form (not flats)
#
# * Apply filtering to streets: not main roads, but sufficiently wide, no overhead power lines, etc
#
# * Some street geometries have a z-coordinate - need to investigate why / whether it's important (and whether it is breaking anything)
#
# * Average distance to UPRN becomes messed up if there is a faraway home that is joined to the street - realistically it wouldn't be connected. Better to filter UPRNs by minimum distance to a street at the start
#
# * Consider linearity of streets - how does this affect Kensa?
#
# * Are streets the right unit? If two streets are connected to form a straight line, should they be considered as one? Note that in the above method street segments are only subsets of one street
#
# * Fix issue with IllegalArgumentException breaking full USRN dataset import
#
# * Scaling the method to all of the UK
#
# * Identifying suitable parameters for suitable length, density, average distance to UPRN - and radius of buffer circles (can we work with Kensa to find an approximate cost function?)
#
# * Used `explode` above to break MultiLineStrings into individual LineStrings assuming these are connected components - but not sure this is true (e.g. for cycles)

# %% [markdown]
#
