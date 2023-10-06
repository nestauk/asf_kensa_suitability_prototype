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
import pandas
from matplotlib import pyplot
from scipy.stats import burr12, gaussian_kde
import numpy
from matplotlib_venn import venn3

# %% [markdown]
# ## Aim
#
# In this notebook we will aim to demonstrate the potential to pre-qualify small -areas (LSOAs for convenience) according to three dimensions - density, tenure and ability to pay.
#
# Density is the LSOA aggregated measure of UPRNs per USRN km.
# Tenure is based on the proportion of owner occupiers in the LSOA.
# Ability to pay is the proportion of the LSOA earning over a given amount.
#
# There are broadly two approaches we can take methodologically - we can use a binary threshold to filter, meaning LSOAs are either 'in' or 'out', or we can score/rank the LSOAs and create a continuous representation of pre-qualification suitability.
#
# In this prototype, for convenience we'll trial a binary approach. However, key to a binary approach is the use of meaningful thresholds. Identifying these would be a task for future work, so for now we'll use some arbitrary candidate values.
#
# Finally, as the Scottish Census hasn't been published yet and the incomes data only covers England and Wales, England and Wales will be the focus of the filtering. Future work will have to look at how to integrate Scotland.

# %%
# Processed street density data path
street_data = "../../inputs/data/lsoa/uprn_street_density_lsoa_2021.csv"

# lsoa polygons from https://geoportal.statistics.gov.uk/maps/766da1380a3544c5a7ca9131dfd4acb6
lsoa_path = "../../inputs/data/lsoa/LSOA_Dec_2021_Boundaries_Generalised_Clipped_EW_BGC_V2_-6143458911719931546.gpkg"

# Tenure data
tenure_path = "../../inputs/data/tables/TS054-2021-4-filtered-2023-07-19T08_48_38Z.csv"

# Income data
income = (
    "../../inputs/data/tables/experimentalabisoccupiedaddresstaxyearending2018.xlsx"
)

# Income distribution fits
income_fits_path = "../../outputs/tables/lsoa_income_fits.parquet"

# Lsoa lookup
lookup = "../../inputs/data/tables/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Lookup_for_England_and_Wales_(Version_2).csv"

# %%
# Density data
density = pandas.read_csv(street_data)

# Lsoas
lsoa = geopandas.read_file(lsoa_path)

# Get eng/wales tenure data
tenure = pandas.read_csv(tenure_path)

# Income
income = pandas.read_excel(income, sheet_name="Net Occupied Address LSOA", skiprows=2)

# Income fits
income_fits = pandas.read_parquet(income_fits_path)

# lsoa lookup
lookup = pandas.read_csv(lookup)

# %%
# Derive tenure stat - owner_occupier_prop
tenure = (
    tenure.pivot(
        index="Lower layer Super Output Areas Code",
        columns="Tenure of household (9 categories)",
        values="Observation",
    )
    .assign(total=lambda df: df.sum(axis=1))
    .reset_index()
    .rename_axis(None, axis=1)
    .assign(
        owner_occupier_prop=lambda df: df[
            ["Owned: Owns outright", "Owned: Owns with a mortgage or loan"]
        ].sum(axis=1)
        / df["total"]
    )
)


# %%
def get_prop(row: pandas.Series, value: float) -> float:
    k, s, scale = row["burrxii_params"]
    prop = burr12(k, s, scale=scale).cdf(value)
    return 1 - prop


# merge income and params, create some income thresholds
income = (
    pandas.concat([income, income_fits["burrxii_params"]], axis=1)
    .assign(
        prop_20k=lambda df: df.apply(lambda row: get_prop(row, 20_000), axis=1),
        prop_30k=lambda df: df.apply(lambda row: get_prop(row, 30_000), axis=1),
        prop_40k=lambda df: df.apply(lambda row: get_prop(row, 40_000), axis=1),
        prop_50k=lambda df: df.apply(lambda row: get_prop(row, 50_000), axis=1),
        prop_60k=lambda df: df.apply(lambda row: get_prop(row, 60_000), axis=1),
        prop_70k=lambda df: df.apply(lambda row: get_prop(row, 70_000), axis=1),
        prop_80k=lambda df: df.apply(lambda row: get_prop(row, 80_000), axis=1),
    )
    .merge(lookup[["LSOA11CD", "LSOA21CD"]], left_on="LSOA code", right_on="LSOA11CD")
    .groupby("LSOA21CD", as_index=False)
    .agg(
        {
            "prop_20k": "mean",
            "prop_30k": "mean",
            "prop_40k": "mean",
            "prop_50k": "mean",
            "prop_60k": "mean",
            "prop_70k": "mean",
            "prop_80k": "mean",
        }
    )
)

# %%
# Merge Data
lsoa = (
    lsoa.merge(density, on="LSOA21NM")
    .merge(tenure, left_on="LSOA21CD", right_on="Lower layer Super Output Areas Code")
    .merge(income, on="LSOA21CD")
)

# %% [markdown]
# ## Some Univariate Statistics
#
# ### Average Street Density
#

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

xs = numpy.linspace(0, 1000, 1001)
z = gaussian_kde(
    lsoa.loc[lsoa["average_street_density"] < 1000, "average_street_density"]
)(xs)

ax.plot(xs, z)
ax.set_xlabel("Mean Linear Density (UPRNS/USRN km)")
ax.set_ylabel("Density")
ax.grid()
ax.set_title("Distribution of Linear Density, LSOAs, England and Wales")

# %%
# Look instead at percentiles
percentiles = numpy.linspace(0, 1, 1001)
values = lsoa["average_street_density"].quantile(percentiles)

f, ax = pyplot.subplots(figsize=(8, 5))

ax.plot(percentiles, numpy.log(values))
ax.axvline(0.131, linestyle="dashed", ymax=0.42)
ax.axvline(0.947, linestyle="dashed", ymax=0.67)
ax.axhline(4.165358494623148, linestyle="dashed", xmax=0.165)  # 64.41577101220506
ax.axhline(5.966119839252738, linestyle="dashed", xmax=0.9)  # 389.98950919142936

ax.set_xlabel("LSOA percentile, England And Wales")
ax.set_ylabel("Log Mean Linear Density (UPRNs/USRN km)")
ax.set_title("Turning points in the LSOA distribution of linear density")


# %%
def turning_points(array):
    """turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and
    maximum turning points in two separate lists.
    """
    idx_max, idx_min = [], []
    if len(array) < 3:
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)

    def get_state(a, b):
        if a < b:
            return RISING
        if a > b:
            return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max


# %%
turning_points(
    numpy.log(values.to_list())
    - (numpy.log(values.to_list()).max() * numpy.arange(0.00, 1.001, 0.001))
)

# %%
lsoa["average_street_density"].between(64, 390).sum()

# %% [markdown]
# ### Owner Occupiers

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

xs = numpy.linspace(0, 1, 1001)
z = gaussian_kde(lsoa["owner_occupier_prop"])(xs)

ax.plot(xs, z)
ax.set_xlabel("Owner Occupier Proportion")
ax.set_ylabel("Density")
ax.grid()
ax.set_title("Distribution of Owner Occupier Proportions, LSOAs, England and Wales")

# %% [markdown]
# ### Proportion at Income Thresholds

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

# 20k
xs = numpy.linspace(0, 1, 1001)
z = gaussian_kde(lsoa["prop_20k"])(xs)
mode20 = xs[numpy.argmax(z)]
ax.plot(xs, z, color="#e41a1c", label=">£20k")

# 30k
xs = numpy.linspace(0, 1, 1001)
z = gaussian_kde(lsoa["prop_30k"])(xs)
mode30 = xs[numpy.argmax(z)]
ax.plot(xs, z, color="#377eb8", label=">£30k")

# 40k
xs = numpy.linspace(0, 1, 1001)
z = gaussian_kde(lsoa["prop_40k"])(xs)
mode40 = xs[numpy.argmax(z)]
ax.plot(xs, z, color="#4daf4a", label=">£40k")

# 50k
xs = numpy.linspace(0, 1, 1001)
z = gaussian_kde(lsoa["prop_50k"])(xs)
mode50 = xs[numpy.argmax(z)]
ax.plot(xs, z, color="#984ea3", label=">£50k")

# 60k
xs = numpy.linspace(0, 1, 1001)
z = gaussian_kde(lsoa["prop_60k"])(xs)
mode60 = xs[numpy.argmax(z)]
ax.plot(xs, z, color="#ff7f00", label=">£60k")

# 70k
# xs = numpy.linspace(0, 1, 1001)
# z = gaussian_kde(lsoa['prop_70k'])(xs)
# ax.plot(xs, z, color='#ffff33', label='£70k')

# 80k
# xs = numpy.linspace(0, 1, 1001)
# z = gaussian_kde(lsoa['prop_80k'])(xs)
# ax.plot(xs, z, color='#a65628', label='£80k')

ax.legend()
ax.set_xlabel("Income Proportion")
ax.set_ylabel("Density")
ax.grid()
ax.set_title("Distribution of Income Proportions, LSOAs, England and Wales")

# %%
mode20, mode30, mode40, mode50, mode60

# %% [markdown]
# ### Multivariate Analysis

# %%
# basic filters
density_filter = lsoa["average_street_density"].between(64, 390)
tenure_filter = lsoa["owner_occupier_prop"] > 0.5
income_filter = lsoa["prop_30k"] > 0.5

# %%
(density_filter & tenure_filter & income_filter).sum()

# %%
f, ax = pyplot.subplots(figsize=(7, 6))

venn3(
    subsets=(
        (density_filter & ~(tenure_filter | income_filter)).sum(),
        (tenure_filter & ~(density_filter | income_filter)).sum(),
        (tenure_filter & density_filter & ~income_filter).sum(),
        (income_filter & ~(tenure_filter | density_filter)).sum(),
        (~tenure_filter & density_filter & income_filter).sum(),
        (tenure_filter & ~density_filter & income_filter).sum(),
        (density_filter & tenure_filter & income_filter).sum(),
    ),
    set_labels=(
        "Density Filter\n(64 - 390 UPRNs/km)",
        "Tenure Filter\n(>50% owner occupier)",
        "Income Filter\n(>50% £30k+)",
    ),
    ax=ax,
)

# %%
numpy.sum(
    (
        (density_filter & ~(tenure_filter | income_filter)).sum(),
        (tenure_filter & ~(density_filter | income_filter)).sum(),
        (tenure_filter & density_filter & ~income_filter).sum(),
        (income_filter & ~(tenure_filter | density_filter)).sum(),
        (~tenure_filter & density_filter & income_filter).sum(),
        (tenure_filter & ~density_filter & income_filter).sum(),
        (density_filter & tenure_filter & income_filter).sum(),
    )
)

# %%
lsoa[(density_filter & tenure_filter & income_filter)]["total"].sum()

# %%
count_dense = []
for val in range(64, 391):
    # basic filters
    density_filter = lsoa["average_street_density"].between(val, 390)
    tenure_filter = lsoa["owner_occupier_prop"] > 0.5
    income_filter = lsoa["prop_30k"] > 0.5
    count_dense.append((density_filter & tenure_filter & income_filter).sum())

count_tenure = []
for val in numpy.linspace(0.5, 1, 51):
    # basic filters
    density_filter = lsoa["average_street_density"].between(64, 390)
    tenure_filter = lsoa["owner_occupier_prop"] > val
    income_filter = lsoa["prop_30k"] > 0.5
    count_tenure.append((density_filter & tenure_filter & income_filter).sum())

count_income = []
for val in numpy.linspace(0.5, 1, 51):
    # basic filters
    density_filter = lsoa["average_street_density"].between(64, 390)
    tenure_filter = lsoa["owner_occupier_prop"] > 0.5
    income_filter = lsoa["prop_30k"] > val
    count_income.append((density_filter & tenure_filter & income_filter).sum())

# %%
f, [ax1, ax2, ax3] = pyplot.subplots(1, 3, figsize=(12, 3), sharey=True)

ax1.plot(range(64, 391), count_dense)
ax2.plot(numpy.linspace(0.5, 1, 51), count_tenure)
ax3.plot(numpy.linspace(0.5, 1, 51), count_income)

ax1.set_xlabel("Linear Property Density")
ax2.set_xlabel("Owner Occupier Proportion")
ax3.set_xlabel("£30k+ Income Proportion")
ax1.set_ylabel("Count of LSOAs")

ax2.set_title("Filter Sensitivity Holding Other Criteria Constant")

# %%
eng_wal = (
    geopandas.read_file(
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Countries_December_2022_GB_BGC/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
    )
    .loc[lambda df: df["CTRY22NM"].isin(["England", "Wales"])]
    .to_crs(27700)
)

# %%
f, [ax1, ax2, ax3] = pyplot.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

# ax1 - density
lsoa.loc[density_filter].plot(color="#ff9999", edgecolor="none", zorder=2, ax=ax1)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax1)
ax1.set_axis_off()
ax1.set_title("Density (64 - 390 UPRNs/km)")

# ax2 - Owner Occupier
lsoa.loc[tenure_filter].plot(color="#99cc99", edgecolor="none", zorder=2, ax=ax2)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax2)
ax2.set_axis_off()
ax2.set_title("Owner Occupiers (>50%)")

# ax3 - Income Filter
lsoa.loc[income_filter].plot(color="#9999ff", edgecolor="none", zorder=2, ax=ax3)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax3)
ax3.set_axis_off()
ax3.set_title("£30k+ Net Household Income (>50%)")


# %%
# calculate venn classes

lsoa.loc[
    lambda df: (density_filter & ~(tenure_filter | income_filter)), "venn_class"
] = 1
lsoa.loc[
    lambda df: (tenure_filter & ~(density_filter | income_filter)), "venn_class"
] = 2
lsoa.loc[lambda df: (tenure_filter & density_filter & ~income_filter), "venn_class"] = 3
lsoa.loc[
    lambda df: (income_filter & ~(tenure_filter | density_filter)), "venn_class"
] = 4
lsoa.loc[lambda df: (~tenure_filter & density_filter & income_filter), "venn_class"] = 5
lsoa.loc[lambda df: (tenure_filter & ~density_filter & income_filter), "venn_class"] = 6
lsoa.loc[lambda df: (density_filter & tenure_filter & income_filter), "venn_class"] = 7

# %%
colors = {
    1: "#ff9999",
    2: "#99cc99",
    3: "#e0bc99",
    4: "#9999ff",
    5: "#da95da",
    6: "#99bce0",
    7: "#c1adc1",
}

# %%
f, [ax1, ax2] = pyplot.subplots(1, 2, figsize=(12, 7))

# ax1
(
    lsoa.plot(
        facecolor=lsoa["venn_class"].map(colors).fillna("0.9"),
        edgecolor=lsoa["venn_class"].map(colors).fillna("0.9"),
        linewidth=0.5,
        zorder=2,
        ax=ax1,
    )
)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax1)
ax1.set_axis_off()

# ax2
(
    lsoa.loc[lambda df: df["venn_class"] == 7].plot(
        facecolor="#c1adc1", edgecolor="#c1adc1", linewidth=0.5, zorder=2, ax=ax2
    )
)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax2)
ax2.set_axis_off()

# %% [markdown]
# ## Ranking within Selected Region

# %%
rank_lsoa = lsoa.loc[lambda df: df["venn_class"] == 7]

# %%
# Range standardise the inputs
rank_lsoa["std_average_street_density"] = (
    rank_lsoa["average_street_density"] - rank_lsoa["average_street_density"].min()
) / (
    rank_lsoa["average_street_density"].max()
    - rank_lsoa["average_street_density"].min()
)
rank_lsoa["std_owner_occupier_prop"] = (
    rank_lsoa["owner_occupier_prop"] - rank_lsoa["owner_occupier_prop"].min()
) / (rank_lsoa["owner_occupier_prop"].max() - rank_lsoa["owner_occupier_prop"].min())
rank_lsoa["std_prop_30k"] = (rank_lsoa["prop_30k"] - rank_lsoa["prop_30k"].min()) / (
    rank_lsoa["prop_30k"].max() - rank_lsoa["prop_30k"].min()
)

# %%
pandas.plotting.scatter_matrix(
    rank_lsoa[
        ["std_average_street_density", "std_owner_occupier_prop", "std_prop_30k"]
    ],
    figsize=(8, 8),
)

# %%
rank_lsoa[
    ["std_average_street_density", "std_owner_occupier_prop", "std_prop_30k"]
].corr(method="pearson")

# %%
rank_lsoa["rank_criteria"] = (
    rank_lsoa["std_average_street_density"] + rank_lsoa["std_prop_30k"]
).rank(ascending=False)

# %%
f, ax = pyplot.subplots(figsize=(8, 12))

rank_lsoa.plot("rank_average_street_density", cmap="Reds_r", zorder=2, ax=ax)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax)
ax.set_axis_off()

# add colorbar
cax = f.add_axes([0.2, 0.42, 0.025, 0.3])
sm = pyplot.cm.ScalarMappable(cmap="Reds_r", norm=pyplot.Normalize(vmin=1, vmax=7740))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
cbar = f.colorbar(sm, cax=cax)
cax.set_title("Rank")
cbar.ax.invert_yaxis()

# %%
f, ax = pyplot.subplots(figsize=(8, 12))

rank_lsoa.plot("rank_prop_30k", cmap="Greens_r", zorder=2, ax=ax)
eng_wal.plot(facecolor="0.9", edgecolor="none", zorder=1, ax=ax)
ax.set_axis_off()

# add colorbar
cax = f.add_axes([0.2, 0.42, 0.025, 0.3])
sm = pyplot.cm.ScalarMappable(cmap="Greens_r", norm=pyplot.Normalize(vmin=1, vmax=7740))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
cbar = f.colorbar(sm, cax=cax)
cax.set_title("Rank")
cbar.ax.invert_yaxis()
