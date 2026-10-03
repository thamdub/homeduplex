# Third-party notices

homeduplex installs these packages at runtime. Each keeps its own license; the full texts ship inside each
installed package (`*.dist-info/licenses/`).

| Package | License | Source |
|---|---|---|
| pydantic, pydantic-core | MIT | https://github.com/pydantic/pydantic |
| PyYAML | MIT | https://github.com/yaml/pyyaml |
| websockets | BSD-3-Clause | https://github.com/python-websockets/websockets |
| numpy | BSD-3-Clause (and bundled permissive licenses) | https://github.com/numpy/numpy |
| wyoming | MIT | https://github.com/OHF-Voice/wyoming |
| httpx, httpcore | BSD-3-Clause | https://github.com/encode/httpx |
| soxr (python-soxr, with libsoxr) | LGPL-2.1-or-later | https://github.com/dofuuz/python-soxr |
| tzdata | Apache-2.0 | https://github.com/python/tzdata |
| anyio, h11, annotated-types, typing-inspection | MIT | their PyPI pages |
| idna | BSD-3-Clause | https://github.com/kjd/idna |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| typing-extensions | PSF-2.0 | https://github.com/python/typing_extensions |

## soxr (LGPL-2.1-or-later)

homeduplex uses soxr, unmodified, as a separately installed library for audio resampling. Under the LGPL you may
replace it with your own build: install a different `soxr` into the same Python environment. Its source code is
available at https://github.com/dofuuz/python-soxr (and libsoxr at https://sourceforge.net/projects/soxr/). The
Docker image includes the soxr package as published on PyPI, with its license file.
