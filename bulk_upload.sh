#!/bin/zsh

WORKDIR=$(dirname "$0")
# Execute the Python script with all provided arguments
python3 ${WORKDIR}/bulk_upload.py "$@"
