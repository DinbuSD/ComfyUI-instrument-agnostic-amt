Vendored code: instrument_agnostic_amt
=====================================

This directory contains a vendored copy of the
`instrument-agnostic-amt` package (MIT License), so that the ComfyUI nodes
work without cloning the upstream repository.

Upstream: https://github.com/anime-song/instrument-agnostic-amt
License:  MIT, Copyright (c) 2026 anime-song — full text in ./LICENSE
          (also at https://github.com/anime-song/instrument-agnostic-amt/blob/main/LICENSE)

Update procedure: replace this directory with the `instrument_agnostic_amt/`
folder from a fresh clone of the upstream repository (keep this README and
the LICENSE file):

    git clone https://github.com/anime-song/instrument-agnostic-amt.git
    cp -r instrument-agnostic-amt/instrument_agnostic_amt <this directory>/

The nodes only use the inference parts. As of upstream commit cdf0ed3
(refactor: reorganize AMT code into packages), training code lives in the
top-level recipes/ directory and is not vendored here.
