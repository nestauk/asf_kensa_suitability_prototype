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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import boto3

s3 = boto3.client("s3")

# %%
# Bucket Name
bucket = "asf-kensa-suitability-prototype"

# %%
# Prior Objects
[obj["Key"] for obj in s3.list_objects(Bucket=bucket)["Contents"]]

# %% [markdown]
# ## Linked Identifiers
#
# Source: https://osdatahub.os.uk/downloads/open/LIDS

# %%
lids = "../../inputs/data/lids/BLPU_UPRN_Street_USRN_11.csv"

# %%
s3.upload_file(
    Filename=lids, Bucket=bucket, Key="data/lids/BLPU_UPRN_Street_USRN_11.csv"
)

# %% [markdown]
# ## LSOAs
#
# Sourced from ONS geoportal and Scottish Government.

# %%
lsoa_21 = "../../inputs/data/lsoa/LSOA_2021_EW_BSC_1975544170032131431.gpkg"
lsoa_21_gen = "../../inputs/data/lsoa/LSOA_Dec_2021_Boundaries_Generalised_Clipped_EW_BGC_V2_-6143458911719931546.gpkg"
lsoa_11 = "../../inputs/data/lsoa/LSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3_-1005832519865330139.gpkg"
scot_lsoa_11 = "../../inputs/data/lsoa/SG_DataZoneBdry_2011.zip"

# %%
s3.upload_file(
    Filename=lsoa_21,
    Bucket=bucket,
    Key="data/lsoa/LSOA_2021_EW_BSC_1975544170032131431.gpkg",
)

# %%
s3.upload_file(
    Filename=lsoa_21_gen,
    Bucket=bucket,
    Key="data/lsoa/LSOA_Dec_2021_Boundaries_Generalised_Clipped_EW_BGC_V2_-6143458911719931546.gpkg",
)

# %%
s3.upload_file(
    Filename=lsoa_11,
    Bucket=bucket,
    Key="data/lsoa/LSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3_-1005832519865330139.gpkg",
)

# %%
s3.upload_file(
    Filename=scot_lsoa_11, Bucket=bucket, Key="data/lsoa/SG_DataZoneBdry_2011.zip"
)

# %% [markdown]
# ## OS OpenMap Local
#
# Source: https://osdatahub.os.uk/downloads/open/OpenMapLocal

# %%
openmap = "../../inputs/data/os_openmap_local/Data/opmplc_gb.gpkg"

# %%
s3.upload_file(
    Filename=openmap, Bucket=bucket, Key="data/os_openmap_local/opmplc_gb.gpkg"
)

# %% [markdown]
# ## Tables
#
# IMD England and Wales estimates - https://www.gov.uk/government/statistics/indices-of-deprivation-2019-income-and-employment-domains-combined-for-england-and-wales
#
# Experimental income data - https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/articles/adminbasedincomestatisticsenglandandwales/taxyearending2018
#
# lsoa lookup - from ONS geoportal
#
# Census Tables - from ONS Census site

# %%
imd_ew = "../../inputs/data/tables/2019_Income_and_Employment_Domains_-_England_and_Wales.ods"
income = (
    "../../inputs/data/tables/experimentalabisoccupiedaddresstaxyearending2018.xlsx"
)
lsoa_lookup = "../../inputs/data/tables/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Lookup_for_England_and_Wales_(Version_2).csv"
rural_urban_classification = "../../inputs/data/tables/Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods"
census_table_1 = "../../inputs/data/tables/custom-filtered-2023-07-20T12_47_48Z.csv"
census_table_2 = (
    "../../inputs/data/tables/TS054-2021-4-filtered-2023-07-19T08_48_38Z.csv"
)

# %%
s3.upload_file(
    Filename=imd_ew,
    Bucket=bucket,
    Key="data/tables/2019_Income_and_Employment_Domains_-_England_and_Wales.ods",
)

# %%
s3.upload_file(
    Filename=income,
    Bucket=bucket,
    Key="data/tables/experimentalabisoccupiedaddresstaxyearending2018.xlsx",
)

# %%
s3.upload_file(
    Filename=lsoa_lookup,
    Bucket=bucket,
    Key="data/tables/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Lookup_for_England_and_Wales_(Version_2).csv",
)

# %%
s3.upload_file(
    Filename=rural_urban_classification,
    Bucket=bucket,
    Key="data/tables/Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods",
)

# %%
s3.upload_file(
    Filename=census_table_1,
    Bucket=bucket,
    Key="data/tables/custom-filtered-2023-07-20T12_47_48Z.csv",
)

# %%
s3.upload_file(
    Filename=census_table_2,
    Bucket=bucket,
    Key="data/tables/TS054-2021-4-filtered-2023-07-19T08_48_38Z.csv",
)

# %% [markdown]
# ## Urban Form
#
# Source: https://urbangrammarai.xyz/blog/post28_proceedings.html

# %%
urban_form = "../../inputs/data/urban_form/signatures_form_simplified.gpkg"

# %%
s3.upload_file(
    Filename=urban_form,
    Bucket=bucket,
    Key="data/urban_form/signatures_form_simplified.gpkg",
)
