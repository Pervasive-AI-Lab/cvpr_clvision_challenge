#!/bin/bash
set -euo pipefail
echo "Clearing filesystem cache"
echo "Executing as root"
sysctl -w vm.drop_caches=3
