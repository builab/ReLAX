#!/usr/bin/env bash

# Get the directory where this script is located
RELAX="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export variables
export RELAX


# Add script directory to PATH if needed
export PATH=$PATH:$RELAX