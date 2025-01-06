#!/bin/zsh

WORKDIR=$(dirname "$0")

# Execute the Python script with all provided arguments
export PYTHONPATH=${WORKDIR}
python3 ${WORKDIR}/scripts/bulk_upload.py "$@"
