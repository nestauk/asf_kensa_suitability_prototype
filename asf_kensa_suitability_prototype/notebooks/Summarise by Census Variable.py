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
from pyogrio import read_dataframe
import pandas
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# %%
# lsoa polygons from https://geoportal.statistics.gov.uk/maps/766da1380a3544c5a7ca9131dfd4acb6
lsoa_path = "../../inputs/data/lsoa/LSOA_Dec_2021_Boundaries_Generalised_Clipped_EW_BGC_V2_-6143458911719931546.gpkg"
# datazone polygons from
datazone_path = "/vsizip/../../inputs/data/lsoa/SG_DataZoneBdry_2011.zip"

# %%
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
lsoa.shape

# %%
lsoa.head()

# %%
# read density data
density_data = pandas.read_csv(
    "../../inputs/data/lsoa/uprn_street_density_lsoa_2021.csv"
)

# %%
density_data.head()

# %%
# Get eng/wales tenure data
tenure = pandas.read_csv(
    "../../inputs/data/tables/TS054-2021-4-filtered-2023-07-19T08_48_38Z.csv"
)

# %%
# reshape tenure data
tenure = (
    tenure.pivot(
        columns="Tenure of household (9 categories)",
        index=["Lower layer Super Output Areas Code", "Lower layer Super Output Areas"],
        values="Observation",
    )
    .assign(total=lambda df: df.sum(axis=1).values)
    .reset_index()
    .rename_axis(None, axis=1)
)
tenure.head()

# %%
# NB current drops scotland due to tenure merge
lsoa = lsoa.merge(density_data, on="LSOA21NM", how="left").merge(
    tenure.drop(columns="Lower layer Super Output Areas"),
    left_on="LSOA21CD",
    right_on="Lower layer Super Output Areas Code",
)
lsoa.head()

# %% [markdown]
# ## Validation Against Households

# %%
lsoa[["clipped_usrn_uprn_estimate", "total"]].corr()

# %%
lsoa.loc[
    lambda df: df["clipped_usrn_uprn_estimate"] <= 2000,
    ["clipped_usrn_uprn_estimate", "total"],
].corr()

# %%
# Get coefficients of linear fit
beta, alpha = numpy.polyfit(
    lsoa.loc[
        lambda df: df["clipped_usrn_uprn_estimate"] <= 2000,
        "clipped_usrn_uprn_estimate",
    ],
    lsoa.loc[lambda df: df["clipped_usrn_uprn_estimate"] <= 2000, "total"],
    1,
)
alpha, beta

# %%
# Quick R2 calculation
yhat = numpy.poly1d([beta, alpha])(
    lsoa.loc[lambda df: df["clipped_usrn_uprn_estimate"] <= 2000, "total"]
)
ybar = lsoa.loc[
    lambda df: df["clipped_usrn_uprn_estimate"] <= 2000, "clipped_usrn_uprn_estimate"
].mean()
ssres = ((yhat - ybar) ** 2).sum()
sstot = (
    (
        lsoa.loc[
            lambda df: df["clipped_usrn_uprn_estimate"] <= 2000,
            "clipped_usrn_uprn_estimate",
        ]
        - ybar
    )
    ** 2
).sum()
r2 = ssres / sstot
r2

# %%
f, ax = plt.subplots()

ax.scatter(
    lsoa.loc[
        lambda df: df["clipped_usrn_uprn_estimate"] <= 2000,
        "clipped_usrn_uprn_estimate",
    ],
    lsoa.loc[lambda df: df["clipped_usrn_uprn_estimate"] <= 2000, "total"],
    marker=".",
    s=6,
    color="k",
    alpha=0.1,
)

ax.plot(
    [100, 1900],
    [(alpha + beta * 100), (alpha + beta * 1900)],
    color="darkorange",
    label=f"Best fit, $R^{2}$={round(r2,2)}",
)

ax.set_ylim([0, 1750])
ax.set_xlim([0, 2000])
ax.set_xlabel("UPRN counts")
ax.set_ylabel("Census 2021 Households")
ax.legend()
# ax.set_title("LSOA-level comparison of UPRN and Census 2021 Household Counts")

# %%
lsoa.loc[lambda df: df["clipped_usrn_uprn_estimate"] <= 2000, "residuals"] = (
    lsoa.loc[lambda df: df["clipped_usrn_uprn_estimate"] <= 2000, "total"] - yhat
)

# %%
f, ax = plt.subplots(figsize=(8, 12))

max_residual = max(lsoa["residuals"].max(), abs(lsoa["residuals"].min()))

norm = TwoSlopeNorm(vcenter=0, vmin=-max_residual, vmax=max_residual)

lsoa.plot("residuals", norm=norm, cmap="coolwarm", ax=ax)

ax.set_axis_off()

# add colorbar
cax = f.add_axes([0.75, 0.55, 0.02, 0.2])
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
# fake up the array of the scalar mappable. Urgh...
sm._A = []
cax.set_title("Residuals")
f.colorbar(sm, cax=cax)

ax2 = f.add_axes([0.1, 0.6, 0.25, 0.13])

ax2.hist(lsoa["residuals"], bins=40, density=True)
ax2.set_title("Histogram of Residuals")
ax2.set_xlim([-200, 450])

# %% [markdown]
# ## Tenure

# %%
lsoa = lsoa.assign(
    own_prop=lsoa[["Owned: Owns outright", "Owned: Owns with a mortgage or loan"]].sum(
        axis=1
    )
    / lsoa["total"],
    private_prop=lsoa[
        [
            "Private rented: Other private rented",
            "Private rented: Private landlord or letting agency",
        ]
    ].sum(axis=1)
    / lsoa["total"],
    social_prop=lsoa[
        [
            "Social rented: Rents from council or Local Authority",
            "Social rented: Other social rented",
        ]
    ].sum(axis=1)
    / lsoa["total"],
)
lsoa.head()

# %%
# create rolling percentage window (+/- 5)
owner_occupied = []
private_rent = []
social_rent = []
for lower, centre, upper in zip(
    numpy.arange(-0.05, 1.00, 0.05),
    numpy.arange(0.0, 1.05, 0.05),
    numpy.arange(0.05, 1.10, 0.05),
):
    owner_occupied.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["own_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["own_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["own_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )
    private_rent.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["private_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["private_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["private_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )
    social_rent.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["social_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["social_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["social_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )

# %%
owner_occupied = numpy.array(owner_occupied)
private_rent = numpy.array(private_rent)
social_rent = numpy.array(social_rent)

owner_occupied = numpy.column_stack(
    [
        owner_occupied,
        owner_occupied[:, 1]
        - 1.96 * (owner_occupied[:, 2] / numpy.sqrt(owner_occupied[:, 3])),
        owner_occupied[:, 1]
        + 1.96 * (owner_occupied[:, 2] / numpy.sqrt(owner_occupied[:, 3])),
    ]
)

private_rent = numpy.column_stack(
    [
        private_rent,
        private_rent[:, 1]
        - 1.96 * (private_rent[:, 2] / numpy.sqrt(private_rent[:, 3])),
        private_rent[:, 1]
        + 1.96 * (private_rent[:, 2] / numpy.sqrt(private_rent[:, 3])),
    ]
)

social_rent = numpy.column_stack(
    [
        social_rent,
        social_rent[:, 1] - 1.96 * (social_rent[:, 2] / numpy.sqrt(social_rent[:, 3])),
        social_rent[:, 1] + 1.96 * (social_rent[:, 2] / numpy.sqrt(social_rent[:, 3])),
    ]
)

# %%
f, ax = plt.subplots(figsize=(8, 6))

# Owner Occupier
ax.plot(
    owner_occupied[:, 0],
    owner_occupied[:, 1],
    color="blue",
    linewidth=0.8,
    label="Owner Occupier",
)
ax.fill_between(
    owner_occupied[:, 0],
    owner_occupied[:, 4],
    owner_occupied[:, 5],
    zorder=0,
    color="blue",
    alpha=0.33,
    ec=None,
)

# private rent
ax.plot(
    private_rent[:, 0],
    private_rent[:, 1],
    color="green",
    linewidth=0.8,
    label="Private Rent",
)
ax.fill_between(
    private_rent[:, 0],
    private_rent[:, 4],
    private_rent[:, 5],
    zorder=0,
    color="green",
    alpha=0.33,
    ec=None,
)

# social rent
ax.plot(
    social_rent[:, 0],
    social_rent[:, 1],
    color="darkorange",
    linewidth=0.8,
    label="Social Rent",
)
ax.fill_between(
    social_rent[:, 0],
    social_rent[:, 4],
    social_rent[:, 5],
    zorder=0,
    color="darkorange",
    alpha=0.33,
    ec=None,
)


# Decoration
ax.set_ylim([0, 1300])
ax.set_xticks(numpy.arange(0.0, 1.1, 0.1))
ax.set_xticklabels(numpy.arange(0, 110, 10))
ax.grid()
ax.legend(loc=2)
ax.set_xlabel("Percentage of Households in LSOA")
ax.set_ylabel("Mean Street Density (UPRNs per km)")

# %%
# create rolling percentage window (+/- 5)
owner_occupied = []
private_rent = []
social_rent = []
for lower, centre, upper in zip(
    numpy.arange(-0.05, 1.00, 0.05),
    numpy.arange(0.0, 1.05, 0.05),
    numpy.arange(0.05, 1.10, 0.05),
):
    owner_occupied.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["own_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["own_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["own_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )
    private_rent.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["private_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["private_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["private_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )
    social_rent.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["social_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["social_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["social_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )

# %%
owner_occupied = numpy.array(owner_occupied)
private_rent = numpy.array(private_rent)
social_rent = numpy.array(social_rent)

# %%
f, ax = plt.subplots(figsize=(8, 6))

# Owner Occupier
ax.plot(
    owner_occupied[:, 0],
    owner_occupied[:, 1],
    color="blue",
    linewidth=0.8,
    label="Owner Occupier",
)
ax.fill_between(
    owner_occupied[:, 0],
    owner_occupied[:, 2],
    owner_occupied[:, 3],
    zorder=0,
    color="blue",
    alpha=0.33,
    ec=None,
)

# private rent
ax.plot(
    private_rent[:, 0],
    private_rent[:, 1],
    color="green",
    linewidth=0.8,
    label="Private Rent",
)
ax.fill_between(
    private_rent[:, 0],
    private_rent[:, 2],
    private_rent[:, 3],
    zorder=0,
    color="green",
    alpha=0.33,
    ec=None,
)

# social rent
ax.plot(
    social_rent[:, 0],
    social_rent[:, 1],
    color="darkorange",
    linewidth=0.8,
    label="Social Rent",
)
ax.fill_between(
    social_rent[:, 0],
    social_rent[:, 2],
    social_rent[:, 3],
    zorder=0,
    color="darkorange",
    alpha=0.33,
    ec=None,
)


# Decoration
ax.set_ylim([0, 800])
ax.set_xticks(numpy.arange(0.0, 1.1, 0.1))
ax.set_xticklabels(numpy.arange(0, 110, 10))
ax.grid()
ax.legend(loc=2)
ax.set_xlabel("Percentage of Households in LSOA")
ax.set_ylabel("Median and IQR of Street Density (UPRNs per km)")

# %% [markdown]
# ## Housing Type

# %%
housing_type = pandas.read_csv(
    "../../inputs/data/tables/custom-filtered-2023-07-20T12_47_48Z.csv"
)

# %%
# reshape tenure data
housing_type = (
    housing_type.pivot(
        columns="Accommodation type (5 categories)",
        index=["Lower layer Super Output Areas Code", "Lower layer Super Output Areas"],
        values="Observation",
    )
    .assign(housing_total=lambda df: df.sum(axis=1).values)
    .reset_index()
    .rename_axis(None, axis=1)
)
housing_type.head()

# %%
# NB current drops scotland due to tenure merge
lsoa = lsoa.merge(
    housing_type.drop(columns="Lower layer Super Output Areas"),
    left_on="LSOA21CD",
    right_on="Lower layer Super Output Areas Code",
)
lsoa.head()

# %%
lsoa = lsoa.assign(
    detached_prop=lsoa["Whole house or bungalow: Detached"] / lsoa["housing_total"],
    semi_prop=lsoa["Whole house or bungalow: Semi-detached"] / lsoa["housing_total"],
    terrace_prop=lsoa["Whole house or bungalow: Terraced"] / lsoa["housing_total"],
    flat_prop=lsoa["Flat, maisonette or apartment"] / lsoa["housing_total"],
)
lsoa.head()

# %%
# create rolling percentage window (+/- 5)
detached = []
semi_detached = []
terraced = []
flat = []
for lower, centre, upper in zip(
    numpy.arange(-0.05, 1.00, 0.05),
    numpy.arange(0.0, 1.05, 0.05),
    numpy.arange(0.05, 1.10, 0.05),
):
    detached.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["detached_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["detached_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["detached_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )
    semi_detached.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["semi_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["semi_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["semi_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )
    terraced.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["terrace_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["terrace_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["terrace_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )
    flat.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["flat_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["flat_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["flat_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )

# %%
detached = numpy.array(detached)
semi_detached = numpy.array(semi_detached)
terraced = numpy.array(terraced)
flat = numpy.array(flat)

detached = numpy.column_stack(
    [
        detached,
        detached[:, 1] - 1.96 * (detached[:, 2] / numpy.sqrt(detached[:, 3])),
        detached[:, 1] + 1.96 * (detached[:, 2] / numpy.sqrt(detached[:, 3])),
    ]
)

semi_detached = numpy.column_stack(
    [
        semi_detached,
        semi_detached[:, 1]
        - 1.96 * (semi_detached[:, 2] / numpy.sqrt(semi_detached[:, 3])),
        semi_detached[:, 1]
        + 1.96 * (semi_detached[:, 2] / numpy.sqrt(semi_detached[:, 3])),
    ]
)

terraced = numpy.column_stack(
    [
        terraced,
        terraced[:, 1] - 1.96 * (terraced[:, 2] / numpy.sqrt(terraced[:, 3])),
        terraced[:, 1] + 1.96 * (terraced[:, 2] / numpy.sqrt(terraced[:, 3])),
    ]
)
# Adjust special case
terraced[-1, :] = numpy.array(
    [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan]
)

flat = numpy.column_stack(
    [
        flat,
        flat[:, 1] - 1.96 * (flat[:, 2] / numpy.sqrt(flat[:, 3])),
        flat[:, 1] + 1.96 * (flat[:, 2] / numpy.sqrt(flat[:, 3])),
    ]
)

# %%
f, ax = plt.subplots(figsize=(8, 6))

# Detached
ax.plot(detached[:, 0], detached[:, 1], color="blue", linewidth=0.8, label="Detached")
ax.fill_between(
    detached[:, 0],
    detached[:, 4],
    detached[:, 5],
    zorder=0,
    color="blue",
    alpha=0.33,
    ec=None,
)

# Semi-Detached
ax.plot(
    semi_detached[:, 0],
    semi_detached[:, 1],
    color="green",
    linewidth=0.8,
    label="Semi-Detached",
)
ax.fill_between(
    semi_detached[:, 0],
    semi_detached[:, 4],
    semi_detached[:, 5],
    zorder=0,
    color="green",
    alpha=0.33,
    ec=None,
)

# Terraced
ax.plot(
    terraced[:, 0], terraced[:, 1], color="darkorange", linewidth=0.8, label="Terraced"
)
ax.fill_between(
    terraced[:, 0],
    terraced[:, 4],
    terraced[:, 5],
    zorder=0,
    color="darkorange",
    alpha=0.33,
    ec=None,
)

# Flat, Maisonette or Apartment
ax.plot(
    flat[:, 0],
    flat[:, 1],
    color="0.5",
    linewidth=0.8,
    label="Flat, Maisonette or Apartment",
)
ax.fill_between(
    flat[:, 0], flat[:, 4], flat[:, 5], zorder=0, color="0.5", alpha=0.33, ec=None
)


# Decoration
ax.set_ylim([0, 800])
ax.set_xticks(numpy.arange(0.0, 1.1, 0.1))
ax.set_xticklabels(numpy.arange(0, 110, 10))
ax.grid()
ax.legend(loc=2)
ax.set_xlabel("Percentage of Households in LSOA")
ax.set_ylabel("Mean Street Density (UPRNs per km)")

# %%
# create rolling percentage window (+/- 5)
detached = []
semi_detached = []
terraced = []
flat = []
for lower, centre, upper in zip(
    numpy.arange(-0.05, 1.00, 0.05),
    numpy.arange(0.0, 1.05, 0.05),
    numpy.arange(0.05, 1.10, 0.05),
):
    detached.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["detached_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["detached_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["detached_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )
    semi_detached.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["semi_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["semi_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["semi_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )
    terraced.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["terrace_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["terrace_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["terrace_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )
    flat.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["flat_prop"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["flat_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["flat_prop"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )

# %%
detached = numpy.array(detached)
semi_detached = numpy.array(semi_detached)
terraced = numpy.array(terraced)
flat = numpy.array(flat)

# %%
f, ax = plt.subplots(figsize=(8, 6))

# Detached
ax.plot(detached[:, 0], detached[:, 1], color="blue", linewidth=0.8, label="Detached")
ax.fill_between(
    detached[:, 0],
    detached[:, 2],
    detached[:, 3],
    zorder=0,
    color="blue",
    alpha=0.33,
    ec=None,
)

# Semi-Detached
ax.plot(
    semi_detached[:, 0],
    semi_detached[:, 1],
    color="green",
    linewidth=0.8,
    label="Semi-Detached",
)
ax.fill_between(
    semi_detached[:, 0],
    semi_detached[:, 2],
    semi_detached[:, 3],
    zorder=0,
    color="green",
    alpha=0.33,
    ec=None,
)

# Terraced
ax.plot(
    terraced[:, 0], terraced[:, 1], color="darkorange", linewidth=0.8, label="Terraced"
)
ax.fill_between(
    terraced[:, 0],
    terraced[:, 2],
    terraced[:, 3],
    zorder=0,
    color="darkorange",
    alpha=0.33,
    ec=None,
)

# Flat, Maisonette or Apartment
ax.plot(
    flat[:, 0],
    flat[:, 1],
    color="0.5",
    linewidth=0.8,
    label="Flat, Maisonette or Apartment",
)
ax.fill_between(
    flat[:, 0], flat[:, 2], flat[:, 3], zorder=0, color="0.5", alpha=0.33, ec=None
)

# Decoration
ax.set_ylim([0, 800])
ax.set_xticks(numpy.arange(0.0, 1.1, 0.1))
ax.set_xticklabels(numpy.arange(0, 110, 10))
ax.grid()
ax.legend(loc=2)
ax.set_xlabel("Percentage of Households in LSOA")
ax.set_ylabel("Median and IQR Street Density (UPRNs per km)")

# %% [markdown]
# ## Income Deprivation

# %%
lsoa_2011_2021_lookup = pandas.read_csv(
    "../../inputs/data/tables/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Lookup_for_England_and_Wales_(Version_2).csv"
)
lsoa_2011_2021_lookup.head()

# %%
# Join to lsoa 2021 lookup and average to account for lsoa mergers.
imd = (
    pandas.read_excel(
        "../../inputs/data/tables/2019_Income_and_Employment_Domains_-_England_and_Wales.ods",
        engine="odf",
        sheet_name="Income",
    )
    .merge(
        lsoa_2011_2021_lookup[["LSOA11CD", "LSOA21CD"]],
        left_on="LSOA Code (2011)",
        right_on="LSOA11CD",
    )
    .groupby("LSOA21CD", as_index=False)[
        "Income Domain Rank (where 1 is most deprived)"
    ]
    .mean()
    .assign(
        percentile=lambda df: df["Income Domain Rank (where 1 is most deprived)"]
        / df.shape[0]
        * 100
    )
)
imd.head()

# %%
lsoa = lsoa.merge(imd, on="LSOA21CD")

# %%
# create rolling percentage window (+/- 5)
income = []
for lower, centre, upper in zip(
    numpy.arange(-5, 100, 5), numpy.arange(0, 105, 5), numpy.arange(5, 110, 5)
):
    income.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["percentile"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa.loc[
                lambda df: df["percentile"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa.loc[
                lambda df: df["percentile"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )

# %%
income = numpy.array(income)

income = numpy.column_stack(
    [
        income,
        income[:, 1] - 1.96 * (income[:, 2] / numpy.sqrt(income[:, 3])),
        income[:, 1] + 1.96 * (income[:, 2] / numpy.sqrt(income[:, 3])),
    ]
)

# %%
# create rolling percentage window (+/- 5)
income_q = []
for lower, centre, upper in zip(
    numpy.arange(-5, 100, 5), numpy.arange(0, 105, 5), numpy.arange(5, 110, 5)
):
    income_q.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["percentile"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["percentile"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["percentile"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )

# %%
income_q = numpy.array(income_q)

# %%
f, ax = plt.subplots(figsize=(8, 6))

# Income Deprivation
ax.plot(
    income[:, 0],
    income[:, 1],
    color="blue",
    linewidth=0.8,
    label="Income Deprivation (mean)",
)
ax.fill_between(
    income[:, 0],
    income[:, 4],
    income[:, 5],
    zorder=0,
    color="blue",
    alpha=0.33,
    ec=None,
)

ax.plot(
    income_q[:, 0],
    income_q[:, 1],
    color="green",
    linewidth=0.8,
    label="Income Deprivation (median and IQR)",
)
ax.fill_between(
    income_q[:, 0],
    income_q[:, 2],
    income_q[:, 3],
    zorder=0,
    color="green",
    alpha=0.33,
    ec=None,
)

# Decoration
ax.legend()
ax.set_ylim([0, 300])
ax.set_xticks(numpy.arange(0, 110, 10))
ax.set_xticklabels(numpy.arange(0, 110, 10))
ax.grid()
ax.set_xlabel("Income Deprivation Percentile")
ax.set_ylabel("Mean Street Density (UPRNs per km)")

# %% [markdown]
# ## Rural-Urban Classification

# %%
rural_urban = pandas.read_excel(
    "../../inputs/data/tables/Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods",
    engine="odf",
    sheet_name="LSOA11",
    skiprows=2,
).merge(
    lsoa_2011_2021_lookup[["LSOA11CD", "LSOA21CD"]],
    left_on="Lower Super Output Area 2011 Code",
    right_on="LSOA11CD",
)
rural_urban.head()

# %%
# 124 lsoas are merged.
# We need to figure our which last the 2021 lsoa should belong too.
# Assume trend towards urbanisation.
agg_rural_urban = []

for name, group in rural_urban.groupby("LSOA21CD"):
    if len(group) == 1:
        agg_rural_urban.append(
            [
                name,
                group["Rural Urban Classification 2011 (2 fold)"]
                .values[0]
                .replace("\xa0", ""),
                group["Rural Urban Classification 2011 (10 fold)"]
                .values[0]
                .replace("\xa0", ""),
                group["Rural Urban Classification 2011 code"]
                .values[0]
                .replace("\xa0", ""),
            ]
        )
    if len(group) > 1:
        if len(group["Rural Urban Classification 2011 (2 fold)"].unique()) == 1:
            ruc_2fold = (
                group["Rural Urban Classification 2011 (2 fold)"]
                .unique()[0]
                .replace("\xa0", "")
            )
        else:
            # Assign Urban
            ruc_2fold = "Urban"
        if len(group["Rural Urban Classification 2011 (10 fold)"].unique()) == 1:
            ruc_10fold = (
                group["Rural Urban Classification 2011 (10 fold)"]
                .unique()[0]
                .replace("\xa0", "")
            )
            ruc_code = (
                group["Rural Urban Classification 2011 code"]
                .unique()[0]
                .replace("\xa0", "")
            )
        else:
            # Assign Higher code e.g. D2 > E2, C1 > D1 etc.
            ruc_10fold = (
                group.sort_values("Rural Urban Classification 2011 code")
                .iloc[0]["Rural Urban Classification 2011 (10 fold)"]
                .replace("\xa0", "")
            )
            ruc_code = (
                group.sort_values("Rural Urban Classification 2011 code")
                .iloc[0]["Rural Urban Classification 2011 code"]
                .replace("\xa0", "")
            )

        agg_rural_urban.append([name, ruc_2fold, ruc_10fold, ruc_code])

# %%
agg_rural_urban = pandas.DataFrame(
    data=agg_rural_urban,
    columns=[
        "LSOA21CD",
        "Rural Urban Classification 2011 (2 fold)",
        "Rural Urban Classification 2011 (10 fold)",
        "Rural Urban Classification 2011 code",
    ],
).assign(
    **{
        "Rural Urban Classification 2011 (10 fold)": lambda df: df[
            "Rural Urban Classification 2011 (10 fold)"
        ]
        .map(
            {
                "Rural town and fringein a sparse setting": "Rural town and fringe in a sparse setting"
            }
        )
        .fillna(df["Rural Urban Classification 2011 (10 fold)"])
    }
)

# %%
# merge with lsoas
lsoa = lsoa.merge(agg_rural_urban, on="LSOA21CD")
lsoa.head()

# %%
lsoa.loc[
    lambda df: df["Rural Urban Classification 2011 (2 fold)"] == "Urban",
    "average_street_density",
].mean()

# %%
lsoa.loc[
    lambda df: df["Rural Urban Classification 2011 (2 fold)"] == "Rural",
    "average_street_density",
].mean()

# %%
lsoa.loc[
    lambda df: df["Rural Urban Classification 2011 (2 fold)"] == "Urban",
    "average_street_density",
].isna().sum()

# %%
f, ax = plt.subplots(figsize=(8, 4.5))

ax.boxplot(
    [
        lsoa.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == "Urban",
            "average_street_density",
        ].fillna(0),
        lsoa.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == "Rural",
            "average_street_density",
        ],
    ],
    showfliers=False,
    labels=["Urban", "Rural"],
)

ax.set_ylabel("Mean Street Density (UPRNs per km)")
ax.set_xlabel("Rural-Urban Classification, England and Wales, 2011 (2-fold)")
ax.grid(axis="y")

# %%
# ruc classes
ruc = [
    "Urban major conurbation",
    "Urban minor conurbation",
    "Urban city and town",
    "Urban city and town in a sparse setting",
    "Rural town and fringe",
    "Rural village and dispersed",
    "Rural town and fringe in a sparse setting",
    "Rural village and dispersed in a sparse setting",
][::-1]

ruc_labels = [
    "Urban major conurbation",
    "Urban minor conurbation",
    "Urban city and town",
    "Urban city and town in a sparse setting",
    "Rural town and fringe",
    "Rural village and dispersed",
    "Rural town and fringe in a sparse setting",
    "Rural village and dispersed in a sparse setting",
][::-1]

# %%
f, ax = plt.subplots(figsize=(8, 4.5))

ax.boxplot(
    [
        lsoa.loc[
            lambda df: df["Rural Urban Classification 2011 (10 fold)"] == ruc_class,
            "average_street_density",
        ].fillna(0)
        for ruc_class in ruc
    ],
    vert=False,
    showfliers=False,
    labels=ruc_labels,
)

ax.set_xlabel("Mean Street Density (UPRNs per km)")
ax.set_ylabel("Rural-Urban Classification\nEngland and Wales, 2011 (8-fold)")
ax.grid(axis="x")

# %% [markdown]
# ## Income Distribution

# %%
income_dist = pandas.read_excel(
    "../../inputs/data/tables/experimentalabisoccupiedaddresstaxyearending2018.xlsx",
    sheet_name="Net Occupied Address LSOA",
    skiprows=2,
)

# %%
# merge
lsoa = lsoa.merge(income_dist, left_on="LSOA21CD", right_on="LSOA code")

# %%
lsoa[["50th percentile (£)", "average_street_density"]].corr()

# %%
lsoa = lsoa.assign(
    income_norm=(lsoa["50th percentile (£)"] - lsoa["50th percentile (£)"].min())
    / (lsoa["50th percentile (£)"].max() - lsoa["50th percentile (£)"].min())
)

# %%
# create rolling percentage window (+/- 5)
median_income = []

for lower, centre, upper in zip(
    numpy.arange(-0.05, 1.00, 0.05),
    numpy.arange(0.0, 1.05, 0.05),
    numpy.arange(0.05, 1.10, 0.05),
):
    median_income.append(
        [
            centre,
            lsoa.loc[
                lambda df: df["income_norm"].between(lower, upper),
                "average_street_density",
            ].median(),
            lsoa.loc[
                lambda df: df["income_norm"].between(lower, upper),
                "average_street_density",
            ].quantile(0.25),
            lsoa.loc[
                lambda df: df["income_norm"].between(lower, upper),
                "average_street_density",
            ].quantile(0.75),
        ]
    )

# %%
median_income = numpy.array(median_income)

# %%
f, ax = plt.subplots(figsize=(8, 6))

# Income Deprivation
ax.plot(
    median_income[:, 0],
    median_income[:, 1],
    color="blue",
    linewidth=0.8,
    label="Income Deprivation (mean)",
)
ax.fill_between(
    median_income[:, 0],
    median_income[:, 2],
    median_income[:, 3],
    zorder=0,
    color="blue",
    alpha=0.33,
    ec=None,
)
