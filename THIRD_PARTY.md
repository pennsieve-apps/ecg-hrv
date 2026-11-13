# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed
from PyPI wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence).
No model weights, data or downloads are bundled: every method runs from code. Licences checked on
PyPI and in each project's LICENSE file on 2026-11-13.

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| NeuroKit2 | 0.3.0 | Dominique Makowski and the NeuroKit2 contributors | MIT | https://github.com/neuropsychology/NeuroKit |
| WFDB for Python (wfdb) | 4.3.1 | MIT Laboratory for Computational Physiology | MIT | https://github.com/MIT-LCP/wfdb-python |
| NumPy | 2.5.4 | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | SciPy Developers | BSD-3-Clause | https://scipy.org |
| pandas | 3.1.1 | The pandas Development Team | BSD-3-Clause | https://pandas.pydata.org |
| h5py | 3.16.0 | Andrew Collette and the h5py contributors | BSD-3-Clause | https://github.com/h5py/h5py |
| Matplotlib | 3.11.2 | Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| scikit-learn | 1.9.1 | The scikit-learn developers | BSD-3-Clause | https://scikit-learn.org |
| joblib | 1.6.0 | Gael Varoquaux and the joblib contributors | BSD-3-Clause | https://github.com/joblib/joblib |
| cloudpickle | 3.1.2 | cloudpickle contributors | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle |
| threadpoolctl | 3.8.0 | threadpoolctl contributors | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| narwhals | 2.28.0 | Marco Gorelli and the narwhals contributors | MIT | https://github.com/narwhals-dev/narwhals |
| PyWavelets | 1.10.0 | The PyWavelets Developers | MIT (bundled parts: BSD-3-Clause) | https://github.com/PyWavelets/pywt |
| requests | 2.34.2 | Kenneth Reitz and contributors | Apache-2.0 | https://github.com/psf/requests |
| certifi | 2026.10.23 | Kenneth Reitz | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | Ahmed Tahri | MIT | https://github.com/jawah/charset_normalizer |
| idna | 3.20 | Kim Davies | BSD-3-Clause | https://github.com/kjd/idna |
| urllib3 | 2.8.0 | Andrey Petrov and contributors | MIT | https://github.com/urllib3/urllib3 |
| fsspec | 2026.9.0 | fsspec contributors | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| contourpy | 1.4.0 | ContourPy Developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project | BSD-3-Clause | https://github.com/matplotlib/cycler |
| fonttools | 4.68.0 | Just van Rossum and fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Jeffrey A. Clark and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |
| python-dateutil | 4.0.0 | dateutil contributors | BSD-3-Clause | https://github.com/dateutil/dateutil |
| six | 1.17.0 | Benjamin Peterson | MIT | https://github.com/benjaminp/six |

`processor/resources.py` is shared with the other Pennsieve processor apps (Apache-2.0).

Not bundled, numbers only: the species presets take published values from PhysioZoo (Behar et al.
2018, Table 2 and Table 3, and the numbers in its `Config/qrs.<species>-ecg.yml` files). PhysioZoo's
code is GPL-3.0 and none of it is used or shipped.

Test-only (not in the image): pytest (MIT) and PyNWB (BSD-3-Clause, Neurodata Without Borders) to
check that the synthetic fixture is valid NWB.
