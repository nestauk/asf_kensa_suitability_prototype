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
import fiona
import geopandas
from dask import dataframe as dd
import dask_geopandas
import pandas
from pyogrio import read_info, read_dataframe
import matplotlib.pyplot as plt

# %% [markdown]
# ## Aim
#
# The aim is to use building outlines from OS OpenMap Local to remove the non-building UPRNs in the OS OpenUPRN data.
#
# I have explored the possibility of excluding all important building classes, but find that in numerous cases the mixed-use nature of the built environment means you cannot simply exclude UPRNs based on their intersection with a building footprint. This is particularly true of retail.
#
# It is likely the case that a true accounting of domestic/residential UPRNs would have to be licensed/purchased.

# %%
# filepath = "/vsizip/../../inputs/data/os_openmap_local/opmplc_gpkg_gb.zip/Data/opmplc_gb.gpkg"
filepath = "../../inputs/data/os_openmap_local/Data/opmplc_gb.gpkg"
uprn_path = "../../inputs/data/uprn/osopenuprn_202304.csv"

# %%
fiona.listlayers(filepath)

# %% [markdown]
# ## UPRNS

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
uprn_gdf = uprn_gdf.set_crs("epsg:27700")

# %% [markdown]
# ## Important Buildings
#
# This is a test on the important buildings layer. What it demonstrates is that ~1.4% of the important buildings do not intersect a UPRN. It appears that in several of these cases a UPRN exists, but it lies just outside of the building polygon.

# %%
important_buildings = geopandas.read_file(filepath, layer="important_building")

# %%
important_buildings.head()

# %%
important_buildings["building_theme"].value_counts()

# %%
important_buildings_gdf = dask_geopandas.from_geopandas(
    important_buildings, npartitions=4
)

# %%
important_building_uprns = dask_geopandas.sjoin(
    uprn_gdf, important_buildings_gdf, how="inner", predicate="intersects"
).compute()

# %%
# 229,748 important buildings
important_buildings_gdf.__len__()

# %%
# 454,891 uprns intersecting with important buildings
important_building_uprns.shape[0]

# %%
# 127,060 ids identified - 55%
# Appears that some places have multiple buildings, some without UPRNs
important_building_uprns.groupby("id")["index_right"].count().sort_values(
    ascending=False
)

# %%
# NB the education theme appears to cover university halls of residence too.
important_building_uprns.groupby("building_theme")["index_right"].count().sort_values(
    ascending=False
)

# %%
# Uprns per building per theme
important_building_uprns.groupby(["building_theme", "id"])[
    "index_right"
].count().reset_index().groupby("building_theme")["index_right"].agg(
    ["min", "max", "mean"]
)

# %%
# Filtering by distinctive name (exact match) gives 3,256 missing, 1.4%
# So actually, I think we're removing most of the important building UPRNs.
# NB several of these seem to be where the UPRN is not within the building polygon.
# A fix might be to buffer the points a small amount (~5-10m?) and rerun the intersection.
important_buildings.loc[
    lambda df: ~df["distinctive_name"].isin(
        important_building_uprns["distinctive_name"].unique()
    ),
    :,
]

# %% [markdown]
# ## Buildings and Railway Stations
#
# Railway stations are not identified as important buildings and are only given as point locations in the OS OpenMap Local data. If we wanted to exclude them, we first intersect the station points with the buildings layer, and the the selected buildings with the UPRN data.
#
# As with important buildings, some stations, particularly those on the London Underground can intersect with large numbers of residential UPRNs.

# %%
# 15m features
read_info(filepath, layer="building")

# %%
railway_stations = geopandas.read_file(filepath, layer="railway_station")

# %%
railway_station_polys = []

for skip in range(0, 15_000_000, 500_000):
    buildings = read_dataframe(
        filepath, layer="building", skip_features=skip, max_features=500_000
    )
    railway_station_polys.append(
        geopandas.sjoin(buildings, railway_stations, how="inner", predicate="contains")
    )
    print(skip + 500_000)

# %%
railway_station_polys = geopandas.pd.concat(railway_station_polys).reset_index(
    drop=True
)

# %%
railway_station_polys.to_file(
    "../../inputs/data/os_openmap_local/railways_station_buildings.gpkg", driver="GPKG"
)

# %%
railway_station_polys = dask_geopandas.read_file(
    "../../inputs/data/os_openmap_local/railways_station_buildings.gpkg", npartitions=8
)

# %%
railway_station_polys = railway_station_polys.drop(columns="index_right")

# %%
railway_station_uprns = dask_geopandas.sjoin(
    uprn_gdf, railway_station_polys, how="inner", predicate="within"
).compute()

# %%
# In London there is overlap between tube stations and housing.
railway_station_uprns.__len__()

# %%
railway_station_uprns["UPRN"].unique().__len__()

# %%
railway_station_uprns.drop(columns="geometry").to_csv(
    "../../inputs/data/os_openmap_local/railways_station_building_uprns.csv",
    index=False,
)

# %% [markdown]
# ## Building UPRNs
#
# Dask geopandas produces memory overflows on sjoin and not enough memory to create a spatial partition, hence this approach.
#
# Building feature codes are:
# 15014 - Building
# 15016 - Glasshouse
# 15018 - Air Transport
# 15019 - Education
# 15020 - Medical Care
# 15021 - Road Transport
# 15022 - Water Transport
# 15023 - Emergency Service
# 15024 - Cultural Facility
# 15025 - Religious Buildings
# 15026 - Retail
# 15027 - Sports Or Exercise Facility
# 15028 - Attraction And Leisure
#
# We'll exclude the codes that are very unlikely to occur in mixed use with residential, namely: glasshouse, air transport, education, road transport, water transport, emergency service, religious buildings, medical care.

# %%
building_uprns = []

for skip in range(0, 15_000_000, 500_000):
    buildings = read_dataframe(
        filepath,
        layer="building",
        skip_features=skip,
        max_features=500_000,
        where="feature_code in (15014, 15024, 15026, 15027, 15028)",
    )

    buildings = dask_geopandas.from_geopandas(buildings, npartitions=10)

    building_uprns.append(
        dask_geopandas.sjoin(
            uprn_gdf, buildings, how="inner", predicate="within"
        ).compute()["UPRN"]
    )
    print(skip + 500_000)
    del buildings

# %%
building_uprns = pandas.concat(building_uprns, ignore_index=True)

# %%
building_uprns.to_csv("../../inputs/data/uprn/building_uprns.csv", index=False)

# %%
building_uprns = dd.read_csv("../../inputs/data/uprn/building_uprns.csv")

# %%
uprn_df = uprn_df.merge(building_uprns, how="inner", on="UPRN")

# %%
# 34,320,971
uprn_df["UPRN"].unique().__len__()

# %%
uprn_df = uprn_df.compute()

# %%
uprn_df.to_csv(
    "../../inputs/data/uprn/osopenuprn_202304_osopenmaplocal_buildings.csv", index=False
)
