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

# %% [markdown]
# ### Imports and setup

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
from shapely import unary_union, line_merge
import contextily as cx

# %%
uprn_path = "inputs/data/uprn/osopenuprn_202304.csv"
usrn_path = "inputs/data/usrn/osopenusrn_202305.gpkg"

# %% [markdown]
# Import UPRNs:

# %%
uprn_df = dd.read_csv(uprn_path)
uprn_df["geometry"] = dask_geopandas.points_from_xy(
    uprn_df, "X_COORDINATE", "Y_COORDINATE"
)
uprn_gdf = dask_geopandas.from_dask_dataframe(uprn_df, geometry="geometry")

# %%
uprn = uprn_gdf.compute()

# %% [markdown]
# Import USRNs:

# %%
# this doesn't work - it raises an IllegalArgumentException:

# usrn_gdf = dask_geopandas.read_file(usrn_path, npartitions=4)
# usrn = usrn_gdf.compute()

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
uprn = gpd.clip(uprn, mask=frame)

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
# Note also that some streets are a weird shape! The cluster of roads in the bottom left seems to correspond to one USRN.

# %% [markdown]
# ### Ranking streets by density of UPRNs (number per unit length)

# %%
# add street lengths
joined["length"] = joined["geometry_line"].length

# %%
joined["uprn_count"] = joined.groupby("usrn")["UPRN"].transform("count")
joined["uprn_density"] = joined["uprn_count"] / joined["length"]

# %%
# note: some streets with zero length, filter to avoid trivial cases
joined = joined.loc[joined["length"] > 0]

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
# Note in the map above that there are some areas with a large number of UPRNs without any streets nearby. Potential risk here that streets surrounding these areas "absorb" lots of UPRNs that they are not actually close to (i.e. too far for Kensa to connect from from their nearest street). We can filter by distance to nearest street to avoid this problem. Also we need to remember to filter UPRNs to just domestic properties, and only those that are suitable (i.e. not blocks of flats).

# %% [markdown]
# ### Identifying dense street segments

# %% [markdown]
# One potential method would involve applying kernel density estimation on the snapped UPRNs to get a continuous density measure on the street network. We could then identify high-density areas by filtering to street segments that have density greater than a particular threshold.
#
# An approximation to this is as follows:
# * For each snapped UPRN, get the segment of the street that is less than a given distance from the point
# * For each street, get the union of these segments
# * Find the connected components of this union
# * Calculate the total lengths of these components and the average distance between the component and the UPRNs that are nearest to it
# * 'Suitable' street segments are therefore components that are sufficiently long and have a sufficiently small average distance-to-UPRN
#
# This is a bit like applying KDE with a uniform kernel. If we used KDE then overlapping kernels would be additive and we would get a better picture of how dense street segments are. But we don't really need this if we have a 'minimum allowable' density in mind - we just need to set the 'given distance' according to this value, then the street segments that result from this process will be 'sufficiently dense'.
#
# The limitation is that we need to know this value in advance, whereas if we used KDE with a non-uniform kernel then we'd get a more general output that could be filtered according to different density values. It's also more strict than KDE - in this method, if two snapped UPRNs are more than twice the 'given distance' apart with nothing in between, then the resulting street segments will be disconnected, but if we used KDE then we could choose a kernel that provides more flexibility, potentially leading to segments being joined if there are enough points on either side.
#
# We can further simplify the first step by just intersecting the street with a buffer circle around the snapped point. Under the assumption that streets are fairly linear then this will roughly correspond to the points on the street that are closer to the point than the radius of the circle. This assumption should be checked though given that we know from the above that some streets are non-linear.

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
# * Some street geometries have a z-coordinate - need to investigate why / whether it's important
#
# * Average distance to UPRN becomes messed up if there is a faraway home that is joined to the street - realistically it wouldn't be connected. Better to filter UPRNs by minimum distance to a street at the start
#
# * Consider linearity of streets - how does this affect Kensa?
#
# * Are streets the right unit? If two streets are connected to form a straight line, should they be considered as one?
#
# * Fix issue with IllegalArgumentException breaking full USRN dataset import
#
# * Scaling the method to all of the UK
#
# * Identifying suitable parameters for suitable length, density, average distance to UPRN - and radius of buffer circles

# %% [markdown]
#
