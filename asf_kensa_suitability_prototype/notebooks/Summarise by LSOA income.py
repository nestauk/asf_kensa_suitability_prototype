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
from matplotlib import pyplot
from matplotlib.colors import LogNorm, Normalize
from scipy import optimize
from scipy import stats
import numpy
import geopandas

# %% [markdown]
# ## Rationale
#
# The experimental income data for LSOAs gives us percentiles of the income distribution for each LSOA.
#
# This is good at telling us, for instance, how median income varies geographically.
#
# However, what if we wanted to know the proportion of households that had a certain income? In theory we could do this by interpolating between the percentiles that we have. According to the spatial variation in incomes, we would expect to see differing proportions of each LSOA achieving at least a given income.
#
# Rather than rely on linearly interpolating between percentiles, we can try fitting a distribution and using that. Popular distributions for household incomes include the Weibull and the Burr XII distribution.

# %%
lsoas = pandas.read_excel(
    "../../inputs/data/tables/experimentalabisoccupiedaddresstaxyearending2018.xlsx",
    sheet_name="Net Occupied Address LSOA",
    skiprows=2,
)

# %%
lsoas.iloc[0, range(2, 11)]

# %%
pyplot.plot(lsoas.iloc[0, range(2, 11)], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
pyplot.ylim([0, 1])
pyplot.xlim([0, 200_000])

# %% [markdown]
# ## Test with Normal Distribution
#
# NB - this is a bad fit, which makes sense.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, mu, sigma: stats.norm(mu, sigma).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[55830, 10000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.norm(*fit_params_ppf).ppf(0.001), stats.norm(*fit_params_ppf).ppf(0.999), 500
)
pyplot.plot(
    xs,
    stats.norm(*fit_params_ppf).cdf(xs),
    "b--",
    label=f"fit ppf: $\\mu={fit_params_ppf[0]:.2f}, \\sigma={fit_params_ppf[1]:.2f}$",
)
pyplot.legend()

# %%
pyplot.hist(stats.norm(*fit_params_ppf).rvs(1000))

# %% [markdown]
# ## Weibull Distribution
#
# NB, need to estimate the `weibull_min` which is the standard Weibull distribution. We estimate both the shape parameter `c` and the `scale` parameter (as the pdf is otherwise in the standardised form)
#
# In this case, the fit looks quite good.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, c, scale: stats.weibull_min(c, scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[1.2, 100000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.weibull_min(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.001),
    stats.weibull_min(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.999),
    500,
)
pyplot.plot(
    xs, stats.weibull_min(fit_params_ppf[0], scale=fit_params_ppf[1]).cdf(xs), "b--"
)

# %%
# Histogram
pyplot.hist(
    stats.weibull_min(fit_params_ppf[0], scale=fit_params_ppf[1]).rvs(10000), bins=50
)

# %% [markdown]
# ## Burr XII Distribution
#
# Takes `c` and `d` as shape parameters as well as a `scale` parameter.
#
# Looks good, but that might just be because of the long 'high income' tail.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, c, d, scale: stats.burr12(c, d, scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[1, 1, 100000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.burr12(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).ppf(
        0.001
    ),
    stats.burr12(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).ppf(
        0.999
    ),
    500,
)
pyplot.plot(
    xs,
    stats.burr12(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).cdf(xs),
    "b--",
)

# %%
# Histogram
pyplot.hist(
    stats.burr12(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).rvs(
        10000
    ),
    bins=50,
)

# %% [markdown]
# ## Pareto
#
# Standard pareto requires a single shape parameter `b`. This is also known as the tail index.
#
# Doesn't seem like as good a fit here as other options due to the long tail. However, lots of pareto variants if we wanted to go down this route (probably not).
#
# The issue with the basic pareto is that it is essentially a power law, so the it just declines (unlike lognormal or weibull).

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, b, scale: stats.pareto(b, scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[1, 10000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.pareto(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.001),
    stats.pareto(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.999),
    500,
)
pyplot.plot(xs, stats.pareto(fit_params_ppf[0], scale=fit_params_ppf[1]).cdf(xs), "b--")

# %%
pyplot.hist(
    stats.pareto(fit_params_ppf[0], scale=fit_params_ppf[1]).rvs(10000), bins=50
)

# %% [markdown]
# ## lognormal
#
# Takes the shape parameter `s` and a scaling factor. The shape parameter is the standard deviation of the log of the distribution.
#
# Not that dissimilar to the weibull.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, s, scale: stats.lognorm(s, scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[0.5, 100000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.lognorm(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.001),
    stats.lognorm(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.999),
    500,
)
pyplot.plot(
    xs, stats.lognorm(fit_params_ppf[0], scale=fit_params_ppf[1]).cdf(xs), "b--"
)

# %%
pyplot.hist(
    stats.lognorm(fit_params_ppf[0], scale=fit_params_ppf[1]).rvs(10000), bins=50
)

# %% [markdown]
# ## A Mielke Beta-Kappa / Dagum Distribution
#
# Takes `k` and `s` as shape parameters.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, k, s, scale: stats.mielke(k, s, scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[1, 1, 100000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.mielke(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).ppf(
        0.001
    ),
    stats.mielke(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).ppf(
        0.999
    ),
    500,
)
pyplot.plot(
    xs,
    stats.mielke(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).cdf(xs),
    "b--",
)

# %%
pyplot.hist(
    stats.mielke(fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]).rvs(
        10000
    ),
    bins=50,
)

# %% [markdown]
# ## Fisk / log-logistic Distribution
#
# Just the shape parameter `c`, plus scale.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, c, scale: stats.fisk(c, scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[2, 100000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.fisk(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.001),
    stats.fisk(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(0.999),
    500,
)
pyplot.plot(xs, stats.fisk(fit_params_ppf[0], scale=fit_params_ppf[1]).cdf(xs), "b--")

# %%
pyplot.hist(stats.fisk(fit_params_ppf[0], scale=fit_params_ppf[1]).rvs(10000), bins=50)

# %% [markdown]
# ## Rayleigh Distribution
#
# No shape parameters, aside from scale. As a result, not a great fit.

# %%
# a list of (p,x) tuples, where P(X<x)=p
percentiles = [
    (0.1, 16598),
    (0.2, 27678),
    (0.3, 37330),
    (0.4, 47464),
    (0.5, 55830),
    (0.6, 68664),
    (0.7, 83251),
    (0.8, 105332),
    (0.9, 153458),
]

fit_params_ppf, _fit_covariances = optimize.curve_fit(
    lambda x, scale: stats.rayleigh(scale=scale).ppf(x),
    xdata=[percent for percent, percentile in percentiles],
    ydata=[percentile for percent, percentile in percentiles],
    p0=[100000],
)

# %%
pyplot.scatter(
    [percentile for percent, percentile in percentiles],
    [percent for percent, percentile in percentiles],
    label="given percentiles",
)
xs = numpy.linspace(
    stats.rayleigh(scale=fit_params_ppf[0]).ppf(0.001),
    stats.rayleigh(scale=fit_params_ppf[0]).ppf(0.999),
    500,
)
pyplot.plot(xs, stats.rayleigh(scale=fit_params_ppf[0]).cdf(xs), "b--")


# %% [markdown]
# ## Find the best performing distribution across all LSOAs
#
# How do we evaluate this? Let's start with a simple RSME approach.
#
# I could take a leave-one-out approach to fitting distributions, but there aren't many data points anyway, so I'll ignore for now.
#
# There's probably a bit of an issue here in that LSOAs with greater incomes will have more influence simply because errors can be larger when absolute values themselves are larger.


# %%
def rmse(ydata_obs: list, ydata_pred: list) -> float:
    se = 0
    for i in range(len(ydata_obs)):
        se += (ydata_obs[i] - ydata_pred[i]) ** 2
    mse = se / len(ydata_obs)
    return mse**0.5


def reshape_inputs(row: pandas.Series) -> list:
    indices = row.index[row.index.str.contains("percentile")]
    return row[indices].to_list()


def fit_lognormal(row: pandas.Series, initial_s: float, initial_scale: float) -> tuple:
    ydata = reshape_inputs(row)
    fit_params_ppf, _fit_covariances = optimize.curve_fit(
        lambda x, s, scale: stats.lognorm(s, scale=scale).ppf(x),
        xdata=numpy.arange(0.1, 1, 0.1),
        ydata=ydata,
        p0=[initial_s, initial_scale],
    )
    error = rmse(
        ydata,
        [
            stats.lognorm(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(pct)
            for pct in numpy.arange(0.1, 1, 0.1)
        ],
    )
    return fit_params_ppf, error


def fit_weibull(row: pandas.Series, initial_c: float, initial_scale: float) -> tuple:
    ydata = reshape_inputs(row)
    fit_params_ppf, _fit_covariances = optimize.curve_fit(
        lambda x, c, scale: stats.weibull_min(c, scale=scale).ppf(x),
        xdata=numpy.arange(0.1, 1, 0.1),
        ydata=ydata,
        p0=[initial_c, initial_scale],
    )
    error = rmse(
        ydata,
        [
            stats.weibull_min(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(pct)
            for pct in numpy.arange(0.1, 1, 0.1)
        ],
    )
    return fit_params_ppf, error


def fit_fisk(row: pandas.Series, initial_c: float, initial_scale: float) -> tuple:
    ydata = reshape_inputs(row)
    fit_params_ppf, _fit_covariances = optimize.curve_fit(
        lambda x, c, scale: stats.fisk(c, scale=scale).ppf(x),
        xdata=numpy.arange(0.1, 1, 0.1),
        ydata=ydata,
        p0=[initial_c, initial_scale],
    )
    error = rmse(
        ydata,
        [
            stats.fisk(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(pct)
            for pct in numpy.arange(0.1, 1, 0.1)
        ],
    )
    return fit_params_ppf, error


def fit_pareto(row: pandas.Series, initial_b: float, initial_scale: float) -> tuple:
    ydata = reshape_inputs(row)
    fit_params_ppf, _fit_covariances = optimize.curve_fit(
        lambda x, b, scale: stats.pareto(b, scale=scale).ppf(x),
        xdata=numpy.arange(0.1, 1, 0.1),
        ydata=ydata,
        p0=[initial_b, initial_scale],
    )
    error = rmse(
        ydata,
        [
            stats.pareto(fit_params_ppf[0], scale=fit_params_ppf[1]).ppf(pct)
            for pct in numpy.arange(0.1, 1, 0.1)
        ],
    )
    return fit_params_ppf, error


def fit_burrXII(
    row: pandas.Series, initial_c: float, initial_d: float, initial_scale: float
) -> tuple:
    ydata = reshape_inputs(row)
    try:
        fit_params_ppf, _fit_covariances = optimize.curve_fit(
            lambda x, c, d, scale: stats.burr12(c, d, scale=scale).ppf(x),
            xdata=numpy.arange(0.1, 1, 0.1),
            ydata=ydata,
            p0=[initial_c, initial_d, initial_scale],
            maxfev=10000,
        )
        error = rmse(
            ydata,
            [
                stats.burr12(
                    fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]
                ).ppf(pct)
                for pct in numpy.arange(0.1, 1, 0.1)
            ],
        )
        return fit_params_ppf, error
    except:
        return None, None


def fit_mielke(
    row: pandas.Series, initial_k: float, initial_s: float, initial_scale: float
) -> tuple:
    ydata = reshape_inputs(row)
    try:
        fit_params_ppf, _fit_covariances = optimize.curve_fit(
            lambda x, k, s, scale: stats.mielke(k, s, scale=scale).ppf(x),
            xdata=numpy.arange(0.1, 1, 0.1),
            ydata=ydata,
            p0=[initial_k, initial_s, initial_scale],
            maxfev=10000,
        )
        error = rmse(
            ydata,
            [
                stats.mielke(
                    fit_params_ppf[0], fit_params_ppf[1], scale=fit_params_ppf[2]
                ).ppf(pct)
                for pct in numpy.arange(0.1, 1, 0.1)
            ],
        )
        return fit_params_ppf, error
    except:
        return None, None


# %%
# NB takes ~4 hours
fits = pandas.concat(
    [
        (
            lsoas.apply(
                lambda row: fit_lognormal(row, 1, 100000), axis=1, result_type="expand"
            ).rename(columns={0: "lognorm_params", 1: "lognorm_error"})
        ),
        (
            lsoas.apply(
                lambda row: fit_weibull(row, 1, 100000), axis=1, result_type="expand"
            ).rename(columns={0: "weibull_params", 1: "weibull_error"})
        ),
        (
            lsoas.apply(
                lambda row: fit_fisk(row, 1, 100000), axis=1, result_type="expand"
            ).rename(columns={0: "fisk_params", 1: "fisk_error"})
        ),
        (
            lsoas.apply(
                lambda row: fit_pareto(row, 1, 10000), axis=1, result_type="expand"
            ).rename(columns={0: "pareto_params", 1: "pareto_error"})
        ),
        (
            lsoas.apply(
                lambda row: fit_burrXII(row, 1, 1, 100000), axis=1, result_type="expand"
            ).rename(columns={0: "burrxii_params", 1: "burrxii_error"})
        ),
        (
            lsoas.apply(
                lambda row: fit_mielke(row, 1, 1, 100000), axis=1, result_type="expand"
            ).rename(columns={0: "mielke_params", 1: "mielke_error"})
        ),
    ],
    axis=1,
)

# %%
fits.to_parquet("../../outputs/tables/lsoa_income_fits.parquet")

# %%
fits = pandas.read_parquet("../../outputs/tables/lsoa_income_fits.parquet")

# %%
# The 2 parameter distributions Burr XII and Mielke outperform the single parameter options tested.
fits[
    [
        "lognorm_error",
        "weibull_error",
        "fisk_error",
        "pareto_error",
        "burrxii_error",
        "mielke_error",
    ]
].describe()

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

for col in fits.columns[fits.columns.str.contains("error")]:
    kernel = stats.gaussian_kde(fits[col])
    xs = numpy.linspace(0, 8000, 500)
    ys = kernel(xs)
    ax.plot(xs, ys, label=col.split("_")[0])

ax.legend(title="Distribution")
ax.set_xlabel("RMSE (£)")

# %% [markdown]
# ## Explore Fits
#
# While the Mielke distribution performs the best, some of the parameter estimates cause overflows when estimating the cdf which we need to establish the proportion of an LSOA covered by a given value.
#
# Therefore, we're using the Burr XII distribution which also performs well.

# %%
f, [[ax1, ax2], [ax3, ax4]] = pyplot.subplots(2, 2, figsize=(12, 8))

# ax1
idx, (k, s, scale) = list(
    fits[fits["burrxii_error"].between(400, 430)].sample(1)["burrxii_params"].items()
)[0]

ax1.scatter(
    lsoas.loc[idx, lsoas.columns.str.contains("percentile")],
    numpy.arange(0.1, 1, 0.1),
    label="given percentiles",
)

xs = numpy.linspace(
    stats.burr12(k, s, scale=scale).ppf(0.001),
    stats.burr12(k, s, scale=scale).ppf(0.999),
    500,
)
ax1.plot(xs, stats.burr12(k, s, scale=scale).cdf(xs), "b--")
ax1.text(x=0, y=0.95, s=f"RMSE: {fits.loc[idx, 'burrxii_error'].round(0)}")
ax1.set_title("Mean LSOA fit")

# ax2
idx, (k, s, scale) = list(
    fits[fits["burrxii_error"].between(635, 681)].sample(1)["burrxii_params"].items()
)[0]

ax2.scatter(
    lsoas.loc[idx, lsoas.columns.str.contains("percentile")],
    numpy.arange(0.1, 1, 0.1),
    label="given percentiles",
)

xs = numpy.linspace(
    stats.burr12(k, s, scale=scale).ppf(0.001),
    stats.burr12(k, s, scale=scale).ppf(0.999),
    500,
)
ax2.plot(xs, stats.burr12(k, s, scale=scale).cdf(xs), "b--")
ax2.text(x=0, y=0.95, s=f"RMSE: {fits.loc[idx, 'burrxii_error'].round(0)}")
ax2.set_title("90th %ile LSOA fit")

# ax3
idx, (k, s, scale) = list(
    fits[fits["burrxii_error"].between(1100, 1600)].sample(1)["burrxii_params"].items()
)[0]

ax3.scatter(
    lsoas.loc[idx, lsoas.columns.str.contains("percentile")],
    numpy.arange(0.1, 1, 0.1),
    label="given percentiles",
)

xs = numpy.linspace(
    stats.burr12(k, s, scale=scale).ppf(0.001),
    stats.burr12(k, s, scale=scale).ppf(0.999),
    500,
)
ax3.plot(xs, stats.burr12(k, s, scale=scale).cdf(xs), "b--")
ax3.text(x=0, y=0.95, s=f"RMSE: {fits.loc[idx, 'burrxii_error'].round(0)}")
ax3.set_title("99th %ile LSOA fit")

# ax4
idx, (k, s, scale) = list(
    fits[fits["burrxii_error"] == fits["burrxii_error"].max()]["burrxii_params"].items()
)[0]

ax4.scatter(
    lsoas.loc[idx, lsoas.columns.str.contains("percentile")],
    numpy.arange(0.1, 1, 0.1),
    label="given percentiles",
)

xs = numpy.linspace(
    stats.burr12(k, s, scale=scale).ppf(0.001),
    stats.burr12(k, s, scale=scale).ppf(0.99),
    500,
)
ax4.plot(xs, stats.burr12(k, s, scale=scale).cdf(xs), "b--")
ax4.text(x=0, y=0.95, s=f"RMSE: {fits.loc[idx, 'burrxii_error'].round(0)}")
ax4.set_title("Worst LSOA fit")

# %% [markdown]
# ## Mapping Errors in Burr XII distribution
#
# There is possibly a bit of spatial structure in the errors associated with wealthier lsoas or lsoas witha larger income range.

# %%
# NB 2011 LSOA definition...
lsoa_geoms = geopandas.read_file(
    "../../inputs/data/lsoa/LSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3_-1005832519865330139.gpkg"
)

# %%
eng_wal = geopandas.read_file(
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Countries_December_2022_UK_BUC/FeatureServer/0/query?where=CTRY22NM%20%3D%20'ENGLAND'%20OR%20CTRY22NM%20%3D%20'WALES'&outFields=*&outSR=27700&f=json"
)

# %%
lsoas["burrxii_error"] = fits["burrxii_error"]
lsoas["burrxii_params"] = fits["burrxii_params"]

# %%
lsoa_geoms = lsoa_geoms.merge(lsoas, left_on="LSOA11CD", right_on="LSOA code")

# %%
pyplot.hist(numpy.log(lsoa_geoms["burrxii_error"]), bins=30)

# %%
f, ax = pyplot.subplots(figsize=(8, 12))

norm = LogNorm(
    vmin=lsoa_geoms["burrxii_error"].min(), vmax=lsoa_geoms["burrxii_error"].max()
)
lsoa_geoms.plot("burrxii_error", cmap="Reds", norm=norm, ax=ax)
eng_wal.plot(facecolor="none", edgecolor="0.1", linewidth=0.5, ax=ax)

# add colorbar
cax = f.add_axes([0.2, 0.45, 0.03, 0.3])
sm = pyplot.cm.ScalarMappable(cmap="Reds", norm=norm)
f.colorbar(sm, label="RMSE (£)", cax=cax)

ax.set_axis_off()


# %% [markdown]
# ## Proportion of households above a given value


# %%
def get_prop(row: pandas.Series, value: float) -> float:
    k, s, scale = row["burrxii_params"]
    prop = stats.burr12(k, s, scale=scale).cdf(value)
    return 1 - prop


# %%
lsoa_geoms["hh_inc_38100_prop"] = lsoa_geoms.apply(
    lambda row: get_prop(row, 38_100), axis=1
)
lsoa_geoms["hh_inc_50000_prop"] = lsoa_geoms.apply(
    lambda row: get_prop(row, 50_000), axis=1
)

# %%
f, [ax1, ax2] = pyplot.subplots(1, 2, figsize=(14, 10))

lsoa_geoms.plot("hh_inc_38100_prop", cmap="Greens", ax=ax1)
eng_wal.plot(facecolor="none", edgecolor="0.1", linewidth=0.5, ax=ax1)
ax1.set_axis_off()
ax1.set_title("LSOA Proportion at £38,100 (median income)")
# add colorbar
cax = f.add_axes([0.1, 0.45, 0.015, 0.3])
sm = pyplot.cm.ScalarMappable(
    cmap="Greens",
    norm=Normalize(
        vmin=lsoa_geoms["hh_inc_38100_prop"].min(),
        vmax=lsoa_geoms["hh_inc_38100_prop"].max(),
    ),
)
f.colorbar(sm, label="Proportion", cax=cax)

lsoa_geoms.plot("hh_inc_50000_prop", cmap="Oranges", ax=ax2)
eng_wal.plot(facecolor="none", edgecolor="0.1", linewidth=0.5, ax=ax2)
ax2.set_axis_off()
ax2.set_title("LSOA Proportion at £50,000")
# add colorbar
cax = f.add_axes([0.55, 0.45, 0.015, 0.3])
sm = pyplot.cm.ScalarMappable(
    cmap="Oranges",
    norm=Normalize(
        vmin=lsoa_geoms["hh_inc_50000_prop"].min(),
        vmax=lsoa_geoms["hh_inc_50000_prop"].max(),
    ),
)
f.colorbar(sm, label="Proportion", cax=cax)

# %% [markdown]
# ## Compare with Street Density Data

# %%
density = pandas.read_csv("../../inputs/data/lsoa/uprn_street_density_lsoa_2011.csv")

# %%
lsoa_geoms = lsoa_geoms.merge(density, on="LSOA11NM")

# %%
# create rolling percentage window (+/- 5)
median = []
fiftyk = []

for lower, centre, upper in zip(
    numpy.arange(-0.05, 1.00, 0.05),
    numpy.arange(0.0, 1.05, 0.05),
    numpy.arange(0.05, 1.10, 0.05),
):
    median.append(
        [
            centre,
            lsoa_geoms.loc[
                lambda df: df["hh_inc_38100_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa_geoms.loc[
                lambda df: df["hh_inc_38100_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa_geoms.loc[
                lambda df: df["hh_inc_38100_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )
    fiftyk.append(
        [
            centre,
            lsoa_geoms.loc[
                lambda df: df["hh_inc_50000_prop"].between(lower, upper),
                "average_street_density",
            ].mean(),
            lsoa_geoms.loc[
                lambda df: df["hh_inc_50000_prop"].between(lower, upper),
                "average_street_density",
            ].std(),
            lsoa_geoms.loc[
                lambda df: df["hh_inc_50000_prop"].between(lower, upper),
                "average_street_density",
            ].count(),
        ]
    )

# %%
median = numpy.array(median)
fiftyk = numpy.array(fiftyk)

median = numpy.column_stack(
    [
        median,
        median[:, 1] - 1.96 * (median[:, 2] / numpy.sqrt(median[:, 3])),
        median[:, 1] + 1.96 * (median[:, 2] / numpy.sqrt(median[:, 3])),
    ]
)

fiftyk = numpy.column_stack(
    [
        fiftyk,
        fiftyk[:, 1] - 1.96 * (fiftyk[:, 2] / numpy.sqrt(fiftyk[:, 3])),
        fiftyk[:, 1] + 1.96 * (fiftyk[:, 2] / numpy.sqrt(fiftyk[:, 3])),
    ]
)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

# Median
ax.plot(
    median[:, 0],
    median[:, 1],
    color="green",
    linewidth=0.8,
    label="Over median Income",
)

ax.fill_between(
    median[:, 0],
    median[:, 4],
    median[:, 5],
    zorder=0,
    color="green",
    alpha=0.33,
    ec=None,
)

# Fifty k
ax.plot(
    fiftyk[:, 0],
    fiftyk[:, 1],
    color="orange",
    linewidth=0.8,
    label="Over £50k Income",
)

ax.fill_between(
    fiftyk[:, 0],
    fiftyk[:, 4],
    fiftyk[:, 5],
    zorder=0,
    color="orange",
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

# %% [markdown]
# ## Comparison to IMD2019

# %%
imd = pandas.read_excel(
    "../../inputs/data/tables/2019_Income_and_Employment_Domains_-_England_and_Wales.ods",
    engine="odf",
    sheet_name="Income",
)

# %%
lsoa_geoms = lsoa_geoms.merge(imd, left_on="LSOA11CD", right_on="LSOA Code (2011)")

# %%
lsoa_geoms[["hh_inc_38100_prop", "Income Domain Rank (where 1 is most deprived)"]].corr(
    method="pearson"
)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

ax.scatter(
    lsoa_geoms["hh_inc_38100_prop"],
    lsoa_geoms["Income Domain Rank (where 1 is most deprived)"],
    marker=".",
    color="k",
    alpha=0.05,
)

ax.set_xlabel("Proportion of LSOAs with Annual Net Household Income >= £38,100")
ax.set_ylabel("IMD 2019 Income Domain Rank, 1 = Most Deprived")
