# AWS F1 bring-up guide — AstraCore Neo NPU

End-to-end recipe for getting our NPU running on an AWS F1 instance.
From zero (no AWS account) to "FPGA runs YOLOv8" in ~1 day of calendar
time, ~6-12 hours of build, ~$50-100 in AWS charges for the first run.

## What you (the user) have to do

These steps need your AWS credentials / billing / console access —
I (Claude) cannot execute them on your behalf.

### 1 · AWS account + permissions (~15 min, one-time)

- If you don't have an AWS account: sign up at https://aws.amazon.com
  (requires a credit card; first 12 months have free-tier credits
  that partially offset F1 costs).
- Create an IAM user with the following managed policies attached:
  - `AmazonEC2FullAccess` (launch F1 instances)
  - `AmazonS3FullAccess` (store AFI tarballs)
  - `AWSMarketplaceManageSubscriptions` (subscribe to the FPGA
    Developer AMI)
- Store the access key + secret locally; you'll need them for the
  AWS CLI steps below.

### 2 · Subscribe to FPGA Developer AMI (~5 min, one-time)

The AMI pre-installs Vivado + AWS HDK + licenses. Without it you'd
spend days setting up Vivado from scratch.

- Open: https://aws.amazon.com/marketplace/pp/prodview-gimv3gqbpe57k
- Click **Continue to Subscribe** → accept terms (free, but required).
- Note the AMI ID for `us-east-1` (or your region) — you'll use it
  in the launch step.

### 3 · Request F1 instance quota (if needed, ~1 hr-1 day)

New AWS accounts usually have an F1 quota of 0. Request one:
- Open Service Quotas → EC2 → "Running On-Demand F instances" →
  Request 8 vCPUs (enough for `f1.2xlarge`, one 2-vCPU slot).
- Approval is usually automatic within minutes; occasionally 1 day.

### 4 · Launch a BUILD instance (~10 min per session, $0.68/hr)

Vivado synthesis doesn't need F1 hardware, just CPU + RAM. Use a
cheaper `c5.4xlarge` for the build, then separately launch F1 only
for runtime.

From the AWS Console (EC2 → Launch Instances):
- AMI: search "FPGA Developer AMI" (AWS Marketplace → paid-enabled)
- Instance type: `c5.4xlarge` (16 vCPU, 32 GB RAM — fits Vivado)
- Storage: 200 GB gp3 (Vivado work area)
- Security group: allow SSH from your IP
- Key pair: create / select one; keep the .pem file

SSH in:
```
ssh -i your-key.pem centos@<instance-public-ip>
```

### 5 · Clone repo + build AFI (~6-12 hr, fully automated after kickoff)

On the build instance:

```bash
# Source AWS HDK env (sets HDK_DIR, PATH, licence vars)
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source hdk_setup.sh
source shared/bin/setup_hdk_env.sh
export CL_DIR=$HDK_DIR/cl/cl_npu

# Pull our repo
cd ~
git clone https://github.com/MandeepCheema/astracore-neo.git
cd astracore-neo/fpga/aws_f1/cl_npu/build

# Build. Expect 6-12 hours; run in `nohup` or tmux so SSH
# disconnects don't kill it.
nohup make build > build.log 2>&1 &
tail -f build.log
```

On successful completion, the DCP + tarball land in:
```
$CL_DIR/build/checkpoints/to_aws/*.tar
```

### 6 · Submit tarball → AFI (async, 30 min-2 hr)

```bash
# Copy to S3
aws s3 cp $CL_DIR/build/checkpoints/to_aws/*.tar s3://YOUR-BUCKET/

# Create FPGA Image
aws ec2 create-fpga-image \
    --name "AstraCoreNeo-v1" \
    --description "AstraCore Neo NPU v1" \
    --input-storage-location Bucket=YOUR-BUCKET,Key=<tarball.tar> \
    --logs-storage-location Bucket=YOUR-BUCKET,Key=logs/

# Save the returned AfiId (fpga-xxx) + AgfiId (agfi-xxx).
```

Poll for AFI readiness:
```bash
aws ec2 describe-fpga-images --filters Name=afi-id,Values=afi-XXXXX
# State transitions: pending → available (success) or failed.
```

You can terminate the c5.4xlarge now — AFI lives in your account
independently.

### 7 · Launch F1 runtime instance + load AFI (~5 min, $1.65/hr)

```bash
# Start f1.2xlarge (same AMI, same key pair, ingress rules)
# SSH in, then:
sudo fpga-load-local-image -S 0 -I agfi-XXXXXXXX
sudo fpga-describe-local-image -S 0   # confirm loaded
```

### 8 · Run the host driver

```bash
cd ~/astracore-neo
pip3 install numpy onnxruntime
sudo python3 fpga/aws_f1/host/f1_npu_driver.py
# Expect: "device-id: 0x41535452  OK"

# Then run a compiled YOLOv8 program:
sudo python3 -c "
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.quantiser import quantise_model, CALIB_PERCENTILE
from tools.npu_ref.conv_compiler import compile_conv2d
from fpga.aws_f1.host.f1_npu_driver import F1NpuDriver, execute_program
import numpy as np
# Load + quant
g = load_onnx('data/models/yolov8n.onnx')
fuse_silu(g)
cal = np.load('data/calibration/yolov8n_calib.npz')['images']
quantise_model(g, 'data/models/yolov8n.onnx',
    [{'images': cal[i:i+1]} for i in range(cal.shape[0])],
    calibration_method=CALIB_PERCENTILE)
# Compile one conv + run on FPGA
layer = next(L for L in g.layers_of('conv') if L.weights is not None)
x = np.load('data/calibration/bus.npz')['image']
res = compile_conv2d(layer.weights.astype(np.int8),
                     x[:, :, :32, :32].astype(np.int8),
                     n_rows=16, n_cols=16,
                     stride=layer.attrs['stride'],
                     pad=layer.attrs['pad'])
with F1NpuDriver(slot=0) as dev:
    out = execute_program(dev, res.program, n_rows=16, n_cols=16)
    print('FPGA result captured:', len(out), 'AO reads')
"
```

## What's cost summary (first demo)

| Item | Cost |
|---|---|
| c5.4xlarge build (one 12-hr run) | ~$8 |
| S3 storage for tarballs (1 GB × 1 month) | <$0.03 |
| AFI creation | $0 (service fee waived) |
| f1.2xlarge runtime (assume 4 hr debug) | ~$7 |
| **Total first-demo** | **~$15-20** |

For iterative development (multiple builds): budget $50-200/month.

## What I (Claude) have already prepared

- `fpga/aws_f1/cl_npu/design/cl_npu.sv` — CL wrapper for npu_top.
- `fpga/aws_f1/cl_npu/design/axi_lite_regfile.sv` — AXI-Lite register
  file (decodes host pokes into NPU cfg_* signals).
- `fpga/aws_f1/host/f1_npu_driver.py` — Python host driver, mmaps
  the OCL BAR and exposes `run_tile` / `read_ao` / `execute_program`.
- `fpga/aws_f1/cl_npu/build/Makefile` — build script integrates with
  AWS HDK's `aws_build_dcp_from_cl.sh` template.

When you SSH into the build instance and `make build`, the Makefile
clones our CL wrapper into `$HDK_DIR/cl/cl_npu/` (using
`cl_dram_dma` as the chassis), appends our RTL to `encrypt.tcl`, and
kicks off the full Vivado synthesis → place → route → bitstream flow.

## What's still stubbed / TODO

- **AXI master DMA** (reading activation bytes from DDR): the CL
  wrapper currently drives `cl_sh_dma_pcis_*` tied low. First demo
  loads activations via register pokes (slow but works for small
  shapes). Full DMA lands in the next iteration.
- **Multi-shell clock crossing**: the CL runs on the 250 MHz user
  clock by default. If we need a different clock for the NPU
  (timing-driven), add a PLL + async FIFO.
- **Activation-bus width 32 bits**: the regfile only implements the
  first 32 bits of AI data; for N_ROWS > 4 at INT8 we need to widen
  to 128+ bits. For the 16x16 pilot this just works; the 64x64
  production build needs the wider path.

## Quick sanity check you should do BEFORE spending the build time

From any machine (doesn't need AWS):
```bash
cd astracore-neo
# Run the existing 8x8 Verilator build — this is the same RTL the
# F1 build will use, just at a smaller array.
wsl -d Ubuntu-22.04 -- bash tools/run_verilator_npu_top_8x8.sh
# Should end with: [PASS] npu_top 8x8 cocotb tests
```

If that PASSes, the RTL is sound and your F1 build will compile.
If it FAILs, fix it before paying for the build.
