#!/bin/bash
# Installs verilator + cocotb + build deps in WSL Ubuntu-22.04.
# Runs as root via wsl -u root, no sudo needed.
set +e

PAT_A='unattended-upgr'
PAT_B='unattended-upgrade-shutdown'

echo "--- killing unattended ---"
pkill -9 -f "$PAT_A" 2>/dev/null
pkill -9 -f "$PAT_B" 2>/dev/null
sleep 2

echo "--- remaining procs ---"
ps -eo pid,comm | awk '$2 ~ /unat/ {print}'

echo "--- clearing locks ---"
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock
rm -f /var/cache/apt/archives/lock /var/lib/apt/lists/lock
ls /var/lib/dpkg/lock* 2>&1

echo "--- dpkg configure ---"
DEBIAN_FRONTEND=noninteractive dpkg --configure -a 2>&1 | tail -10

echo "--- apt update ---"
DEBIAN_FRONTEND=noninteractive apt-get update -y 2>&1 | tail -10

echo "--- apt install ---"
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    verilator python3-pip make g++ ca-certificates 2>&1 | tail -20

echo "--- verify ---"
which verilator
verilator --version
which python3
python3 --version
which pip3
echo "--- install cocotb ---"
pip3 install cocotb==2.0.1 2>&1 | tail -20
python3 -c "import cocotb; print('cocotb OK', cocotb.__version__)"

echo "--- DONE ---"
