<!-- #region -->

# ASF/Kensa Suitability Prototype

All work has been conducted in notebooks as work is primarily exploratory. Broadly, notebooks cover work as follows:

- UPRN filtering - `OS OpenMap Local UPRN Filtering.ipynb`
  - This notebook explores the use of OS OpenMa local as a way of filtering out non-domestic UPRNs. Success is partial.
- Metric creation - `experiments.ipynb`
  - This notebook covers processing the UPRN, USRN and Linked identifiers data and deriving a USRN-based simple measure of linear density. It also covers some experimental attempts to implement a smoothing-based approach.
- Metric mapping - `LSOA Map.ipynb`
  - Covers aggreagtion to LSOA (and Datazones) and representation of the density metric.
- Exploration of density metric against Census data - `Summarise by Census Variable.ipynb`
  - Summarises density according to some key variables - including tenure, hosuing type and indices of deprivation.
- Exploration of density metric against urban form data - `Summarise by urban typology.ipynb`
  - Summarises density using a model of Urban Form developed by 'Urban Grammar' at the University of Liverpool.
- Exploration of density metric against net household income data - `Summarise by LSOA income.ipynb`
  - Summarises density using experimental ONS income data.
- Experiment with pre-qualifying LSOAs for share gorund arrays - `Prototype LSOA Filter.ipynb`
  - Exploration of some basic filtering and ranking of key variables to identify candidate LSOAs.
- Move data into s3 bucket - `Store Data.ipynb`
  - Convenience code to move data into s3 buckets.

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:

  - Setup the conda environment
  - Configure `pre-commit`

- Download data from S3 and unzip (or download raw [UPRN](https://beta.ordnancesurvey.co.uk/products/os-open-uprn) and [USRN](https://beta.ordnancesurvey.co.uk/products/os-open-usrn) data from OS).
- Additionally download relevant [linked identifier data](https://beta.ordnancesurvey.co.uk/products/os-open-linked-identifiers) choosing the 'BLPU UPRN Street USRN 11' option from the OS Open Data Hub.
- All required data are available in s3.
- Add it to `/inputs` according to the following structure:

```
inputs/
├─ data/
│  ├─ uprn/
│     ├─ osopenuprn_202304.csv
│  ├─ usrn/
│     ├─ osopenuprn_202305.gpkg
│  ├─ lids/
│     ├─ BLPU_UPRN_Street_USRN_11.csv
```

- This structure can be extended as per key in s3 to meet the data requirements of the notebooks.

- Install required packages: `pip install -r requirements.txt`
- Install Jupyter kernel: `ipython kernel install --user --name=asf_kensa_suitability_prototype`
- Open notebook with kernel `asf_kensa_suitability_prototype`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>

<!-- #endregion -->
