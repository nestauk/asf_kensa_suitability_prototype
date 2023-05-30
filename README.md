# ASF/Kensa Suitability Prototype

All work is currently in one notebook: `asf_kensa_suitability_prototype/notebooks/experiments.py`

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:

  - Setup the conda environment
  - Configure `pre-commit`

- Download data from S3 and unzip (or download raw [UPRN](https://beta.ordnancesurvey.co.uk/products/os-open-uprn) and [USRN](https://beta.ordnancesurvey.co.uk/products/os-open-usrn) data from OS). Add it to `/inputs` according to the following structure:

```
inputs/
├─ data/
│  ├─ uprn/
│     ├─ osopenuprn_202304.csv
│  ├─ usrn/
│     ├─ osopenuprn_202305.gpkg
```

- Install required packages: `pip install -r requirements.txt`
- Install Jupyter kernel: `ipython kernel install --user --name=asf_kensa_suitability_prototype`
- Open notebook with kernel `asf_kensa_suitability_prototype`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
