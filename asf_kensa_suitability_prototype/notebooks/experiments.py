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
from rasterio.features import rasterize
import numpy as np

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

# lsoa polygons from https://geoportal.statistics.gov.uk/maps/766da1380a3544c5a7ca9131dfd4acb6
lsoa_path = "inputs/data/lsoa/LSOA_Dec_2021_Boundaries_Generalised_Clipped_EW_BGC_2022_3280346050083114707.gpkg"

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
# note that we still have duplicate geometries
usrn_gdf.drop_duplicates("geometry").__len__()

# relevant info from uprn.uk:
#
# Note: a USRN is an operational identifier and is unique to a highway authority.
# Where a road crosses a highway authority boundary, therefore, it will have different USRNs
# in different authorities, even if it has the same name and number.
#
# Roads operated by national highway authorities (eg, Highways England) will also have
# separate USRNs for the road as a whole and for each subsection within each local highway
# authority that it crosses.

# %% [markdown]
# ### USRN "density"

# %%
# add column indicating density of UPRNs on USRNs
usrn_gdf["density"] = usrn_gdf["count"] / usrn_gdf["geometry"].length

# %%
usrn_gdf["density"].describe().compute()

# %% [markdown]
# ### Summaries by LSOA

# %%
# read LSOA data
lsoa = read_dataframe(lsoa_path)

# %%
lsoa_gdf = dask_geopandas.from_geopandas(lsoa, npartitions=10)

# %%
# overlay usrn_gdf with lsoa_gdf so that each row of clipped_gdf
# corresponds to a segment of a USRN contained within an LSOA
lsoa_gdf["lsoa_geometry"] = lsoa_gdf.geometry
overlaid_gdf = usrn_gdf.compute().overlay(lsoa_gdf.compute(), how="intersection")

# %%
overlaid_gdf = dask_geopandas.from_geopandas(overlaid_gdf, npartitions=10)

# %%
# for each LSOA, calculate average street density by dividing total clipped USRN "weight" (density * clipped length)
# by total length of clipped USRNs
overlaid_gdf["clipped_usrn_length"] = overlaid_gdf["geometry"].length
overlaid_gdf["clipped_usrn_weight"] = (
    overlaid_gdf["clipped_usrn_length"] * overlaid_gdf["density"]
)

# %%
lsoa_summaries = overlaid_gdf.groupby("LSOA21NM")[
    ["clipped_usrn_length", "clipped_usrn_weight"]
].agg("sum")
lsoa_summaries["average_street_density"] = (
    lsoa_summaries["clipped_usrn_weight"] / lsoa_summaries["clipped_usrn_length"]
)

# %%
lsoa_summaries.sort_values("average_street_density").compute()

# %% [markdown]
# The LSOA with lowest density is Cardiff 048F (W01001947) which is an industrial section of Cardiff Bay; other LSOAs in the top 5 are large and rural. The LSOA with highest density is Salford 036B (E01033994) which contains a large block of flats. (Interactive map of LSOAs [here](https://geoportal.statistics.gov.uk/datasets/766da1380a3544c5a7ca9131dfd4acb6))

# %% [markdown]
# ### Rasterisation

# %%
from affine import Affine

# Declare processing extent (bounds in British National Grid)
xmin = 5000
xmax = 666000
ymin = 6000
ymax = 1221000

# Affine transformation for surface in British national grid, with 500m cell size.
cell_size = 500
transform = Affine(cell_size, 0, xmin, 0, -cell_size, ymax)

out_shape = (int((ymax - ymin) / cell_size), int((xmax - xmin) / cell_size))

shapes = [(geom, dens) for geom, dens in zip(usrn_gdf["geometry"], usrn_gdf["density"])]

# %%
raster = rasterize(shapes=shapes, out_shape=out_shape, transform=transform)

# %%
# plot with 1 pixel = 1 cell on 96 dpi screen
dpi = (96, 96)
figsize = (raster.shape[0] / dpi[0], raster.shape[1] / dpi[1])

# %%
# plot nonzero cells
plt.figure(figsize=figsize)
plt.imshow((raster != 0), interpolation="none")

# %%
# cell density values are long-tailed
print((raster > 0.5).sum())
print((raster > 1).sum())
print((raster > 2).sum())
print((raster > 4).sum())
print((raster > 8).sum())

# %%
# scale by taking log of (raster + 1)
# not the easiest to see
plt.figure(figsize=figsize)
plt.imshow(np.log(raster + 1), interpolation="none")

# %%
# better: clip to [0, 1]
plt.figure(figsize=figsize)
plt.imshow(np.clip(raster, 0, 1), interpolation="none")

# %%
# sense check: take a look at the areas corresponding to the highest value raster cells


def plot_nth_largest_raster_cell(n):
    # Flatten the array
    flat_raster = raster.flatten()

    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(flat_raster)[::-1]

    # Get the index of the nth-largest value
    nth_largest_index = sorted_indices[n - 1]

    # Convert the flattened index to the corresponding indices in the original array
    nth_largest_indices = np.unravel_index(nth_largest_index, raster.shape)

    # Create a geometry representing the area of the nth-largest value cell
    x_coord = xmin + nth_largest_indices[1] * cell_size
    y_coord = ymax - nth_largest_indices[0] * cell_size
    max_cell_geometry = box(x_coord, y_coord - cell_size, x_coord + cell_size, y_coord)

    # Plot usrn_gdf clipped to this extent
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_coord, x_coord + cell_size)
    ax.set_ylim(y_coord - cell_size, y_coord)
    usrn_gdf_area = gpd.clip(usrn_gdf.compute(), mask=max_cell_geometry)
    usrn_gdf_area.plot(ax=ax, alpha=0.5)
    cx.add_basemap(ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik)


# %%
plot_nth_largest_raster_cell(1)

# %%
plot_nth_largest_raster_cell(2)


# %%
plot_nth_largest_raster_cell(3)

# %%
plot_nth_largest_raster_cell(raster.size)

# %% [markdown]
# Seems about as expected on the whole, but surprising that so much of the top cell is unoccupied - may be worth exploring further

# %% [markdown]
# ## Point-snapping approach

# %% [markdown]
# Restrict the geometries to a bounded area to make computation manageable (specific area chosen is a section of the east coast between Great Yarmouth and Lowestoft):

# %%
frame = (628160, 290000, 660000, 310000)

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
# * Scaling the method to all of the UK
#
# * Identifying suitable parameters for suitable length, density, average distance to UPRN - and radius of buffer circles (can we work with Kensa to find an approximate cost function?)
#
# * Used `explode` above to break MultiLineStrings into individual LineStrings assuming these are connected components - but not sure this is true (e.g. for cycles)
#
# * How does property density vary along the average street? If it's fairly constant then we don't really need to worry about street segments

# %% [markdown]
# ### Point-snapping approach applied to all GB

# %%
# copied from above

# import fiona
# from shapely.geometry import shape
# # Find bad geoms
# bad_usrns = []
# with fiona.open(usrn_path) as src:
#     for feat in src:
#         try:
#             shape(feat.geometry)
#         except:
#             bad_usrns.append(feat.properties['usrn'])

# %%
usrn_gdf = read_dataframe(
    usrn_path, where=f"usrn not in ({','.join(map(str, bad_usrns))})"
)

usrn_gdf = dask_geopandas.from_geopandas(usrn_gdf, npartitions=10)

# %%
uprn_gdf = dask_geopandas.from_dask_dataframe(
    uprn_df,
    geometry=dask_geopandas.points_from_xy(uprn_df, "X_COORDINATE", "Y_COORDINATE"),
)

# %% [markdown]
# `sjoin_nearest` isn't implemented in dask_geopandas and takes ages. Instead we could use the linked UPRN-USRN dataset to get nearest points on the USRN that each UPRN is linked to. In fact this may be more accurate than the original method - the closest USRN to each UPRN isn't necessarily the most sensible one to connect to, whereas the linked USRN may be more reliable and unlikely to be too far away.

# %%
uprn_gdf = uprn_gdf.set_crs("EPSG:27700")
joined_full = uprn_gdf.merge(
    usrn_gdf,
    how="inner",
    left_on="IDENTIFIER_2",
    right_on="usrn",
    suffixes=["_uprn", "_usrn"],
)

# %%
# tidy up
joined_full = joined_full.drop(
    columns=["X_COORDINATE", "Y_COORDINATE", "IDENTIFIER_1", "IDENTIFIER_2"]
)

# %%
joined_full = joined_full.set_geometry("geometry_usrn")
joined_full["length"] = joined_full["geometry_usrn"].length

# %%
joined_full["uprn_count"] = joined_full.groupby("usrn")["UPRN"].transform("count")
joined_full["uprn_density"] = joined_full["uprn_count"] / joined_full["length"]

# %%
joined_full["nearest_point"] = joined_full.apply(
    lambda x: nearest_points(x["geometry_uprn"], x["geometry_usrn"])[1], axis=1
)

# %%
joined_full["total_uprn_to_usrn_distance"] = joined_full["geometry_uprn"].distance(
    joined_full["geometry_usrn"], align=False
)

# %%
# joined_full.head()

# %% [markdown]
# At this point in the method applied to the small area above we identified buffer circles around each snapped point and merged them to get street segments. This takes too long on the full dataset, so instead we simplify by just aggregating this joined dataset by USRN to get street-level features.

# %%
usrn_agg = joined_full.groupby(["usrn"]).agg(
    {
        "total_uprn_to_usrn_distance": "sum",
        "length": "first",
        "uprn_count": "first",
        "uprn_density": "first",
        "geometry_usrn": "first",
    }
)

# %%
usrn_agg_computed = usrn_agg.compute()

# %%
usrn_agg_computed["average_uprn_to_usrn_distance"] = (
    usrn_agg_computed["total_uprn_to_usrn_distance"] / usrn_agg_computed["uprn_count"]
)

# %% [markdown]
# This takes about an hour to compute on my machine. This is now a dataset of USRNs with the following measures that could be used to assess their suitability:
# - Number of linked properties
# - Length of street
# - Density of linked properties along street
# - Average distance between property and street
#
# If we can filter UPRNs to just eligible domestic properties, and filter USRNs to suitable streets, then these measures should correspond to some of the physical characteristics that are relevant to Kensa. As a general thought, potentially it would be best to use the rasterization approach to identify areas, then apply the more detailed point-snapping approach to these areas to get a more granular picture of the streets within them.

# %%
# questionably large max of 184,686
usrn_agg_computed["average_uprn_to_usrn_distance"].describe()

# %%
# suspiciously all of these roads are on the England/Scotland border (from uprn.uk)

usrn_agg_computed.loc[usrn_agg_computed["average_uprn_to_usrn_distance"] > 100000]

# %%
# as an example, get "top streets" according to arbitrary metrics

top_streets = usrn_agg_computed.loc[
    (usrn_agg_computed["average_uprn_to_usrn_distance"] < 100)
    & (usrn_agg_computed["uprn_density"] > 10)
    & (usrn_agg_computed["length"] > 100)
]

# %%
top_streets

# %%
f, ax = plt.subplots(figsize=(15, 15))
top_streets.set_geometry("geometry_usrn").plot(
    ax=ax, color="red", linewidth=3, alpha=0.5
)
cx.add_basemap(
    ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
)

# %% [markdown]
# Both of the "top streets" happen to be in Manchester - both appear to be located near blocks of student accommodation, though also with lots of other non-domestic buildings around.

# %% [markdown]
# Thought: are USRN geometries accurate enough to give an accurate value for UPRN-USRN distance?

# %%
# code for applying the full point-snapping method to the whole of GB
# not sure how long it takes as it's never finished

# joined_full["buffer"] = joined_full.set_geometry("nearest_point").buffer(50)
# joined_full["segment"] = joined_full["buffer"].intersection(joined_full["geometry_usrn"])

# intersections_full = joined_full.groupby("usrn").apply(
#     lambda x: line_merge(unary_union(x["segment"]))
# )

# intersections_full = intersections_full.reset_index().rename(columns={0: "geometry"})

# exploded_full = intersections_full.set_geometry("geometry").explode()

# exploded_full = exploded_full.set_crs("EPSG:27700")

# segment_counts = (
#     joined_full[["nearest_point", "distance"]]
#     .set_geometry("nearest_point")
#     .sjoin_nearest(exploded_full)
# )

# exploded_full["uprn_count"] = segment_counts.groupby("index_right")["nearest_point"].count()
# exploded_full["total_distance"] = segment_counts.groupby("index_right")["distance"].sum()

# exploded_full["uprn_count"] = exploded_full["uprn_count"].fillna(0).astype(int)
# exploded_full["total_distance"] = exploded_full["total_distance"].fillna(0)

# exploded_full["average_distance"] = exploded_full["total_distance"] / exploded_full["uprn_count"]

# exploded_full["length"] = exploded_full["geometry"].length
# exploded_full["density"] = exploded_full["uprn_count"] / exploded_full["length"]

# fig, ax = plt.subplots(figsize=(15, 15))

# # arbitrary parameters for now
# exploded_full.loc[
#     (exploded_full["length"] > 100)
#     & (exploded_full["density"] > 0.2)
#     & (exploded_full["average_distance"] < 50)
# ].plot(ax=ax, color="red")
# cx.add_basemap(
#     ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5
# )
# ax.set_title(
#     "Street segments with 'sufficient' length, UPRN density and average distance to UPRN"
# )

# %% [markdown]
# ## Other useful datasets
#
# ### Open
#
# * Socio-economic - [admin-based income statistics by LSOA](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/articles/adminbasedincomestatisticsenglandandwales/taxyearending2018)
#
# * Pavements - [link on this page no longer works but could enquire](https://www.esriuk.com/en-gb/news/press-releases/uk/39-map-of-every-pavement-width-in-great-britain)
#
# * Overhead lines/cables - [National Grid network route map](https://www.nationalgrid.com/electricity-transmission/network-and-infrastructure/network-route-mapshttps://www.nationalgrid.com/electricity-transmission/network-and-infrastructure/network-route-maps)
#
# * Outdoor space - [MSOA level average garden size](https://www.ons.gov.uk/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain)
#
# * Road widths - could proxy from UPRN/USRN data, distance between adjacent UPRNs
#
# * Tenure - estimate proportions by small area from EPC
#
# * Floor area - as above
#
# ### Closed
#
# * Road widths - [OS Mastermap Highways?](https://beta.ordnancesurvey.co.uk/products/os-mastermap-highways-network-roads)

# %% [markdown]
#
