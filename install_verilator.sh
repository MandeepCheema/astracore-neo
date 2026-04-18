#!/bin/bash
# Direct install — skip dpkg --configure -a entirely (no broken packages found)
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

echo "=== Updating apt ==="
sudo apt-get update -qq < /dev/null 2>&1 | tail -3

echo "=== Installing verilator ==="
sudo apt-get install -y -o Dpkg::Options::="--force-confold" --no-install-recommends verilator < /dev/null 2>&1 | tail -5

echo "=== Installing python3-pip make g++ ==="
sudo apt-get install -y -o Dpkg::Options::="--force-confold" --no-install-recommends python3-pip make g++ < /dev/null 2>&1 | tail -5

echo "=== Installing cocotb ==="
pip3 install --user --break-system-packages cocotb 2>&1 | tail -3

echo "=== VERIFICATION ==="
verilator --version || echo "VERILATOR FAILED"
python3 -c "import cocotb; print(f'cocotb {cocotb.__version__}')" || echo "COCOTB FAILED"
g++ --version | head -1 || echo "G++ FAILED"
echo "=== DONE ==="
