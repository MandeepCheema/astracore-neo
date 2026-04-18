"""Run cocotb simulations using Python subprocess instead of make."""
import subprocess
import sys
import os

def run_module(module_name):
    """Run simulation for one RTL module."""
    sim_dir = f"sim/{module_name}"
    rtl_file = f"rtl/{module_name}/{module_name}.v"
    test_file = f"test_{module_name}"
    sim_build = f"{sim_dir}/sim_build"
    
    os.makedirs(sim_build, exist_ok=True)
    
    # Step 1: Compile with iverilog
    vvp_out = f"{sim_build}/sim.vvp"
    iverilog_cmd = [
        "iverilog", "-o", vvp_out, "-s", module_name, "-g2012", rtl_file
    ]
    
    env = os.environ.copy()
    env["TOPLEVEL"]   = module_name
    env["MODULE"]     = test_file
    env["TOPLEVEL_LANG"] = "verilog"
    env["SIM"]        = "icarus"
    env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")
    
    # Use cocotb VPI library
    vpi_path = ".venv/Lib/site-packages/cocotb/libs"
    vpi_libs = []
    for f in os.listdir(vpi_path):
        if "icarus" in f.lower() and f.endswith(".vpl"):
            vpi_libs.append(os.path.abspath(f"{vpi_path}/{f}"))
    
    if not vpi_libs:
        # Try .vpi extension
        for f in os.listdir(vpi_path):
            if "icarus" in f.lower():
                vpi_libs.append(os.path.abspath(f"{vpi_path}/{f}"))
    
    print(f"\n{'='*60}")
    print(f"Running: {module_name}")
    print(f"VPI libs: {vpi_libs}")
    
    # Compile
    result = subprocess.run(iverilog_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"COMPILE ERROR:\n{result.stderr}")
        return False
    print(f"Compiled OK: {vvp_out}")
    
    # Run vvp with VPI
    vvp_cmd = ["vvp", "-M", vpi_path]
    for lib in vpi_libs:
        vvp_cmd.extend(["-m", lib.replace("\\", "/").split("/")[-1].replace(".vpl","").replace(".vpi","")])
    vvp_cmd.append(vvp_out)
    
    print(f"Run: {' '.join(vvp_cmd)}")
    result = subprocess.run(vvp_cmd, capture_output=True, text=True, env=env, cwd=sim_dir)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    return result.returncode == 0

if __name__ == "__main__":
    module = sys.argv[1] if len(sys.argv) > 1 else "thermal_zone"
    run_module(module)
