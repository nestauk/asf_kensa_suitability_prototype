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
import geopandas
from pyogrio import read_dataframe
import pandas
import numpy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# %% [markdown]
# ## Map Urban Form Data

# %%
# read typology data
# available from: https://urbangrammarai.xyz/blog/post28_proceedings.html
form = read_dataframe(
    "../../inputs/data/urban_form/signatures_form_simplified.gpkg"
).to_crs(27700)

# %%
form.signature_type.unique()

# %%
# Signature Types
#
# https://strathprints.strath.ac.uk/80527/1/Fleischmann_Bel_ISUF_2021_Classifying_urban_form_at_national_scale.pdf
#
# 0 - Countryside
# 1 - Suburban low density development
# 2 - Residential Neighbourhoods
#    2.1 -
#    2.2 -
#    2.3 -
#    2.4 -
#    2.5 -
#    2.6 -
# 3 - Countryside
# 4 - Dense City Centres
#    4.1 -
#    4.2 -
#    4.3 -
#    4.4 -
#    4.5 -
#    4.6 -
#    4.7 -
#    4.8 -
# 5 - Countryside
# 6 - Countryside
# 7 - Countryside

form.signature_type.value_counts()

# %%
london = geopandas.read_file(
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Regions_December_2022_EN_BFC/FeatureServer/0/query?where=RGN22NM%20%3D%20'LONDON'&outFields=*&outSR=27700&f=json"
)
west_midlands = geopandas.read_file(
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Combined_Authorities_December_2022_EN_BFC/FeatureServer/0/query?where=CAUTH22NM%20%3D%20'WEST%20MIDLANDS'&outFields=*&outSR=27700&f=json"
)

# %%
fig = plt.figure(figsize=(10, 8))

gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[3])

# ax1
form.loc[lambda df: df["signature_type"].str.contains("[03567]_\d", regex=True)].plot(
    facecolor="#66c2a5", edgecolor="#66c2a5", ax=ax1
)
form.loc[lambda df: df["signature_type"].str.contains("[12]_\d", regex=True)].plot(
    facecolor="#8da0cb", edgecolor="#8da0cb", ax=ax1
)
form.loc[lambda df: df["signature_type"].str.contains("4_\d", regex=True)].plot(
    facecolor="#fc8d62", edgecolor="#fc8d62", ax=ax1
)

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Great Britain")

# ax2
minx, miny, maxx, maxy = london.total_bounds
form.loc[lambda df: df["signature_type"].str.contains("[03567]_\d", regex=True)].plot(
    facecolor="#66c2a5", edgecolor="#66c2a5", ax=ax2
)
form.loc[lambda df: df["signature_type"].str.contains("[12]_\d", regex=True)].plot(
    facecolor="#8da0cb", edgecolor="#8da0cb", ax=ax2
)
form.loc[lambda df: df["signature_type"].str.contains("4_\d", regex=True)].plot(
    facecolor="#fc8d62", edgecolor="#fc8d62", ax=ax2
)
london.plot(facecolor="none", edgecolor="k", zorder=5, linewidth=0.8, ax=ax2)
ax2.set_xlim([minx - 1500, maxx + 1500])
ax2.set_ylim([miny - 1500, maxy + 1500])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Greater London")

# ax3
minx, miny, maxx, maxy = west_midlands.total_bounds
form.loc[lambda df: df["signature_type"].str.contains("[03567]_\d", regex=True)].plot(
    facecolor="#66c2a5", edgecolor="#66c2a5", ax=ax3
)
form.loc[lambda df: df["signature_type"].str.contains("[12]_\d", regex=True)].plot(
    facecolor="#8da0cb", edgecolor="#8da0cb", ax=ax3
)
form.loc[lambda df: df["signature_type"].str.contains("4_\d", regex=True)].plot(
    facecolor="#fc8d62", edgecolor="#fc8d62", ax=ax3
)
west_midlands.plot(facecolor="none", edgecolor="k", zorder=5, linewidth=0.8, ax=ax3)
ax3.set_xlim([minx - 1500, maxx + 1500])
ax3.set_ylim([miny - 1500, maxy + 1500])
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_title("West Midlands")

plt.subplots_adjust(wspace=0, hspace=0)

# Add legend
handles = [Patch(color="#66c2a5"), Patch(color="#8da0cb"), Patch(color="#fc8d62")]
labels = ["Countryside", "Suburban", "City Centres"]
ax1.legend(handles, labels, loc="upper left")

# %%
cmap = ListedColormap(
    [
        "none",
        "#e7e1ef",
        "#d4b9da",
        "#c994c7",
        "#df65b0",
        "#e7298a",
        "#ce1256",
        "#980043",
        "#67001f",
        "#3f007d",
        "#54278f",
        "#6a51a3",
        "#807dba",
        "#9e9ac8",
        "#bcbddc",
        "#dadaeb",
        "#efedf5",
        "#fcfbfd",
    ]
)

norm = BoundaryNorm(boundaries=numpy.linspace(-0.5, 16.5, 18), ncolors=18)

recode_signatures = {
    "2_0": 1,
    "2_1": 2,
    "2_2": 3,
    "2_3": 4,
    "2_4": 5,
    "2_5": 6,
    "2_6": 7,
    "2_7": 8,
    "4_0": 9,
    "4_1": 10,
    "4_2": 11,
    "4_3": 12,
    "4_4": 13,
    "4_5": 14,
    "4_6": 15,
    "4_7": 16,
    "4_8": 17,
}

# %%
manchester = geopandas.read_file(
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Combined_Authorities_December_2022_EN_BFC/FeatureServer/0/query?where=CAUTH22NM%20%3D%20'GREATER%20MANCHESTER'&outFields=*&outSR=27700&f=json"
)

# %%
f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 10))

# ax1
minx, miny, maxx, maxy = london.total_bounds
form.loc[lambda df: df["signature_type"].str.contains("[03567]_\d", regex=True)].plot(
    facecolor="0.67", edgecolor="0.67", ax=ax1
)
form.loc[lambda df: df["signature_type"].str.contains("1_0", regex=True)].plot(
    facecolor="#f7f4f9", edgecolor="#f7f4f9", ax=ax1
)
(
    form.loc[lambda df: df["signature_type"].str.contains("[24]_\d", regex=True)]
    .assign(
        signature_code=lambda df: df["signature_type"].map(recode_signatures).fillna(0)
    )
    .plot("signature_code", cmap=cmap, norm=norm, ax=ax1)
)
london.plot(facecolor="none", edgecolor="k", zorder=5, linewidth=0.8, ax=ax1)
ax1.set_xlim([minx - 1500, maxx + 1500])
ax1.set_ylim([miny - 1500, maxy + 1500])
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Greater London")

# ax2
minx, miny, maxx, maxy = west_midlands.total_bounds
form.loc[lambda df: df["signature_type"].str.contains("[03567]_\d", regex=True)].plot(
    facecolor="0.67", edgecolor="0.67", ax=ax2
)
form.loc[lambda df: df["signature_type"].str.contains("1_0", regex=True)].plot(
    facecolor="#f7f4f9", edgecolor="#f7f4f9", ax=ax2
)
(
    form.loc[lambda df: df["signature_type"].str.contains("[24]_\d", regex=True)]
    .assign(
        signature_code=lambda df: df["signature_type"].map(recode_signatures).fillna(0)
    )
    .plot("signature_code", cmap=cmap, norm=norm, ax=ax2)
)
west_midlands.plot(facecolor="none", edgecolor="k", zorder=5, linewidth=0.8, ax=ax2)
ax2.set_xlim([minx - 1500, maxx + 1500])
ax2.set_ylim([miny - 1500, maxy + 1500])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("West Midlands")

# ax3
minx, miny, maxx, maxy = manchester.total_bounds
form.loc[lambda df: df["signature_type"].str.contains("[03567]_\d", regex=True)].plot(
    facecolor="0.67", edgecolor="0.67", ax=ax3
)
form.loc[lambda df: df["signature_type"].str.contains("1_0", regex=True)].plot(
    facecolor="#f7f4f9", edgecolor="#f7f4f9", ax=ax3
)
(
    form.loc[lambda df: df["signature_type"].str.contains("[24]_\d", regex=True)]
    .assign(
        signature_code=lambda df: df["signature_type"].map(recode_signatures).fillna(0)
    )
    .plot("signature_code", cmap=cmap, norm=norm, ax=ax3)
)
manchester.plot(facecolor="none", edgecolor="k", zorder=5, linewidth=0.8, ax=ax3)
ax3.set_xlim([minx - 1500, maxx + 1500])
ax3.set_ylim([miny - 1500, maxy + 1500])
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_title("Greater Manchester")

plt.subplots_adjust(wspace=0.05, hspace=0)

# ax4 - legend
ax4.set_axis_off()

legend1 = plt.legend(
    handles=[Patch(color="0.67"), Patch(color="#f7f4f9")]
    + [Patch(color=c) for c in cmap.colors[1:9]],
    labels=["Countryside", "Suburban Low Density"]
    + [
        "Residential Neighbourhoods " + label.split("_")[1]
        for label in list(recode_signatures.keys())[:8]
    ],
    loc="center left",
    frameon=False,
)
legend2 = plt.legend(
    handles=[Patch(color=c) for c in cmap.colors[9:]],
    labels=[
        "City Centres " + label.split("_")[1]
        for label in list(recode_signatures.keys())[8:]
    ],
    loc="center right",
    frameon=False,
)
ax4.add_artist(legend1)
ax4.add_artist(legend2)

ax4.text(0.01, 0.8, "Urban Form Class", fontsize=14)

# %% [markdown]
# ## Form Summaries

# %%
# usrn data
usrn_gdf = geopandas.read_parquet("../../outputs/vectors/gb_usrn_lids_density.parquet")

# %%
# add column indicating density of UPRNs on USRNs
# NB Calculating density per km for more convenient numbers
usrn_gdf["density"] = usrn_gdf["count"] / (usrn_gdf["geometry"].length / 1_000)

# %%
# overlay usrn_gdf with form geodataframe so that each row of clipped_gdf
# corresponds to a segment of a USRN contained within an form polygon
parts = []
for part in [
    "1_0",
    "2_6",
    "2_5",
    "2_4",
    "2_3",
    "2_2",
    "2_7",
    "2_1",
    "4_0",
    "2_0",
    "4_4",
    "4_2",
    "4_5",
    "4_1",
    "4_7",
    "4_6",
    "4_8",
    "4_3",
]:
    form = read_dataframe(
        "../../inputs/data/urban_form/signatures_form_simplified.gpkg",
        where=f"signature_type == '{part}'",
    ).to_crs(27700)
    form["form_geometry"] = form.geometry  # .to_wkt()
    parts.append(usrn_gdf.overlay(form, how="intersection", keep_geom_type=True))
    print(part)

# %%
overlay = pandas.concat(parts, ignore_index=True)
overlay.head()

# %%
del usrn_gdf, part, parts

# %%
# Creates an absurdly large file - do not use!
# overlay.to_file("../../outputs/vectors/urban_form_streets.gpkg", driver='GPKG')

# %%
# for each LSOA, calculate average street density by dividing total clipped USRN estimate of UPRNs (density * clipped length)
# by total length of clipped USRNs
# NB this assumes that UPRNS are uniformly distributed along USRNs
overlay["clipped_usrn_length_km"] = overlay["geometry"].length / 1000

overlay["clipped_usrn_uprn_estimate"] = (
    overlay["clipped_usrn_length_km"] * overlay["density"]
)

# %%
recode_signatures = {
    "1_0": "Suburban low density development",
    "2_0": "Residential neighbourhoods",
    "2_1": "Residential neighbourhoods",
    "2_2": "Residential neighbourhoods",
    "2_3": "Residential neighbourhoods",
    "2_4": "Residential neighbourhoods",
    "2_5": "Residential neighbourhoods",
    "2_6": "Residential neighbourhoods",
    "2_7": "Residential neighbourhoods",
    "4_0": "Dense city centres",
    "4_1": "Dense city centres",
    "4_2": "Dense city centres",
    "4_3": "Dense city centres",
    "4_4": "Dense city centres",
    "4_5": "Dense city centres",
    "4_6": "Dense city centres",
    "4_7": "Dense city centres",
    "4_8": "Dense city centres",
}

overlay["signature_agg"] = overlay["signature_type"].map(recode_signatures)

# %%
# Group by lsoas and derive lsoa totals for uprn count estimates and length of usrns.
form_summaries = overlay.groupby("signature_type")[
    ["clipped_usrn_length_km", "clipped_usrn_uprn_estimate"]
].agg("sum")

# Calculate form-level density
form_summaries["average_street_density"] = (
    form_summaries["clipped_usrn_uprn_estimate"]
    / form_summaries["clipped_usrn_length_km"]
)

# %%
form_summaries

# %%
form_summaries["clipped_usrn_uprn_estimate"] / form_summaries[
    "clipped_usrn_uprn_estimate"
].sum() * 100

# %%
# Group by lsoas and derive lsoa totals for uprn count estimates and length of usrns.
form_agg_summaries = overlay.groupby("signature_agg")[
    ["clipped_usrn_length_km", "clipped_usrn_uprn_estimate"]
].agg("sum")

# Calculate form-level density
form_agg_summaries["average_street_density"] = (
    form_agg_summaries["clipped_usrn_uprn_estimate"]
    / form_agg_summaries["clipped_usrn_length_km"]
)

# %%
form_agg_summaries

# %%
form_agg_summaries["clipped_usrn_uprn_estimate"] / form_agg_summaries[
    "clipped_usrn_uprn_estimate"
].sum()
