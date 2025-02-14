# src/run_analysis.py
from datetime import datetime
from cowstudyapp.config import ConfigManager, AnalysisConfig, HMMConfig, DistributionTypes
from pathlib import Path
import subprocess
import logging
import json
import platform
import os
import sys

def find_r_executable():
    """Find the R executable on the system"""
    if platform.system() == "Windows":
        # Common R installation paths on Windows
        possible_paths = [
            r"C:\Program Files\R\R-4.3.2\bin\Rscript.exe",  # Adjust version as needed
            r"C:\Program Files\R\R-4.3.2\bin\x64\Rscript.exe",
            r"C:\Program Files\R\R-4.3.0\bin\Rscript.exe"
            # Add other possible paths
        ]
        
        # Check R_HOME environment variable
        r_home = os.environ.get("R_HOME")
        if r_home:
            possible_paths.append(Path(r_home) / "bin" / "Rscript.exe")
        
        # Try to find Rscript in PATH
        try:
            result = subprocess.run(["where", "Rscript.exe"], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                possible_paths.extend(result.stdout.splitlines())
        except subprocess.SubprocessError:
            pass

        # Return first existing path
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path))
                
    else:  # Linux/Mac
        try:
            return subprocess.check_output(["which", "Rscript"]).decode().strip()
        except subprocess.SubprocessError:
            pass
    
    raise FileNotFoundError("Could not find R executable. Please install R and ensure it's in your PATH or"
                            "designated in the `r_executable` field in your configuration file.")



def install_r_packages(packages):
    """Install missing R packages"""
    quoted_packages = ", ".join(f"'{pkg.strip()}'" for pkg in packages)
    r_script = f"""
    install.packages(c({quoted_packages}), 
                    repos='https://cran.rstudio.com/')
    """
    
    r_executable = find_r_executable()
    try:
        subprocess.run([r_executable, "--vanilla", "-e", r_script],
                      check=True)
        logging.info(f"Successfully installed R packages: {packages}")
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to install R packages: {e}")


def check_r_packages(r_executable):
    """Check if required R packages are installed"""
    # List of required packages
    required_packages = ["momentuHMM", "dplyr", "ggplot2", "caret", "fitdistrplus",
                 "circular", "CircStats", "lubridate", 'grid', 'gridExtra',
                 "movMF"
    ]
    
    # Step 1: Just get installed packages
    try:
        # First just try to get the list of installed packages
        list_cmd = subprocess.run(
            [r_executable, "-e", "cat(paste(installed.packages()[,1], collapse=','))"],
            capture_output=True,
            text=True,
            check=True
        )
        
        installed_packages = list_cmd.stdout.split(',')
        logging.info(f"Found {len(installed_packages)} installed R packages")
        
        # Step 2: Check which required packages are missing
        missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]
        
        if missing_packages:
            # Ask user if they want to install missing packages
            print(f"\nMissing R packages: {', '.join(missing_packages)}")
            response = input("Would you like to install them? [y/N] ")
            
            if response.lower() == 'y':
                # Install packages one at a time
                for pkg in missing_packages:
                    print(f"\nInstalling {pkg}...")
                    install_cmd = [
                        r_executable,
                        "-e",
                        f"install.packages('{pkg}', repos='https://cran.rstudio.com/')"
                    ]
                    try:
                        subprocess.run(install_cmd, check=True)
                        print(f"Successfully installed {pkg}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to install {pkg}: {e}")
                        raise
            else:
                quoted_packages = ", ".join(f"'{pkg}'" for pkg in missing_packages)
                raise RuntimeError(
                    f"Missing R packages: {', '.join(missing_packages)}\n"
                    f"Please install them by running:\n"
                    f"Rscript -e 'install.packages(c({quoted_packages}))'"
                )
        else:
            logging.info("All required R packages are installed")
            
    except subprocess.CalledProcessError as e:
        # Add more diagnostic information
        error_msg = (
            f"Error running R command:\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"STDOUT: {e.stdout if e.stdout else 'Empty'}\n"
            f"STDERR: {e.stderr if e.stderr else 'Empty'}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)


def stream_r_output(process, prefix="R"):
    """Stream output from an R process in real-time with interrupt handling"""
    try:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                logging.info(f"{prefix}: {line}")
    except KeyboardInterrupt:
        # Terminate the subprocess
        process.terminate()
        # If terminate doesn't work, force kill after a timeout
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise  # Re-raise the KeyboardInterrupt
    finally:
        process.stdout.close()

def get_r_dist_name(dist_type: DistributionTypes) -> str:
    """Convert Python distribution enum to R distribution name"""
    mapping = {
        DistributionTypes.LOGNORMAL: "lnorm",
        DistributionTypes.GAMMA: "gamma",
        DistributionTypes.WEIBULL: "weibull",
        DistributionTypes.NORMAL: "norm",
        DistributionTypes.EXPONENTIAL: "exp",
        DistributionTypes.VONMISES: "vm",
        DistributionTypes.WRAPPEDCAUCHY: "wrpcauchy"
    }
    return mapping[dist_type]


def create_analysis_directories(base_dir: Path, features: list, analysis_type: str) -> dict:
    """
    Create directory structure for HMM analysis outputs
    
    Args:
        base_dir: Base output directory (data/analysis_results/hmm)
        features: List of features being used
        analysis_type: Either 'LOOCV' or 'Product'
    
    Returns:
        Dictionary containing paths to different output directories
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create feature identifier
    feature_names = sorted([f.name for f in features])
    feature_id = "-".join(feature_names)
    if len(feature_id) > 50:  # Limit length
        feature_id = feature_id[:50]
    
    # Create directory structure
    analysis_dir = base_dir / analysis_type / timestamp
    
    # Create subdirectories
    plots_dir = analysis_dir / "plots"
    dist_plots_dir = plots_dir / "distributions"
    models_dir = analysis_dir / "models"  # New directory for saved models
    
    # Create all directories
    for directory in [analysis_dir, plots_dir, dist_plots_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        "base_dir": analysis_dir,
        "plots_dir": plots_dir,
        "dist_plots_dir": dist_plots_dir,
        "models_dir": models_dir
    }

def run_hmm_analysis_old(config: AnalysisConfig, data_path: Path, output_dir: Path):
    """Run the HMM analysis using R scripts"""
    
    r_executable = config.r_executable or find_r_executable()
    check_r_packages(r_executable)
    
    # Ensure R script directory exists
    r_script_dir = Path("src/cowstudyapp/analysis/HMM")
    if not r_script_dir.exists():
        raise FileNotFoundError(f"R script directory not found: {r_script_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert Python config to R format
    r_config = {
        "states": config.hmm.states,
        "features": [feature.model_dump() for feature in config.hmm.features],
        "mode": config.mode,
        "training_info":{
            "training_info_type": config.training_info.training_info_type,
            "training_info_path": config.training_info.training_info_path,
        },
        "options": {
            **config.hmm.options,
            "distributions": {
                "regular": [get_r_dist_name(DistributionTypes(d)) 
                        for d in config.hmm.options["distributions"]["regular"]],
                "circular": [get_r_dist_name(DistributionTypes(d)) 
                            for d in config.hmm.options["distributions"]["circular"]]
            }
        }
    }

    # Save config for R
    config_path = output_dir / "hmm_config.json"
    with open(config_path, "w") as f:
        json.dump(r_config, f, indent=2)
    
    # Convert paths to absolute paths
    paths = {
        'script': r_script_dir / "run_hmm.r",
        'util': r_script_dir / "util.r",  # Add util.r path
        'config': config_path.absolute(),
        'data': data_path.absolute(),
        'output': output_dir.absolute()
    }
    
    # Build R command
    cmd = [
        str(r_executable),
        "--vanilla",
        "--no-save",
        str(paths['script']),
        str(paths['util']),  # Pass util.r path as argument
        str(paths['config']),
        str(paths['data']),
        str(paths['output'])
    ]
    
    try:
        r_executable = find_r_executable()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        stream_r_output(process)
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, terminating R process...")
        if process.poll() is None:  # If process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("R process didn't terminate, forcing kill...")
                process.kill()
        raise



def run_hmm_analysis(config: AnalysisConfig, target_data_path: Path, output_dir: Path):
    """Run the HMM analysis using R scripts"""
    
    r_executable = config.r_executable or find_r_executable()
    check_r_packages(r_executable)
    
    # Ensure R script directory exists
    r_script_dir = Path("src/cowstudyapp/analysis/HMM")
    if not r_script_dir.exists():
        raise FileNotFoundError(f"R script directory not found: {r_script_dir}")
    


    # analysis_type = config.mode if hasattr(config, 'mode') else "LOOCV"

    dirs = create_analysis_directories(
        base_dir=output_dir,
        features=config.hmm.features,
        analysis_type=config.mode
    )

    r_script_path = Path(__file__).parent / "cowstudyapp" / "analysis" / "HMM" / "run_hmm.r"
    util_path = r_script_path.parent / "util.r"


    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert Python config to R format
    r_config = {
        "states": config.hmm.states,
        "features": [feature.model_dump() for feature in config.hmm.features],
        "mode": config.mode,
        "training_info":{
            "training_info_type": config.training_info.training_info_type,
            "training_info_path": config.training_info.training_info_path,
        },
        "options": {
            **config.hmm.options,
            "distributions": {
                "regular": [get_r_dist_name(DistributionTypes(d)) 
                        for d in config.hmm.options["distributions"]["regular"]],
                "circular": [get_r_dist_name(DistributionTypes(d)) 
                            for d in config.hmm.options["distributions"]["circular"]]
            }
        }
    }

    config_path: Path = dirs["base_dir"] / "hmm_config.json"
    with open(config_path, 'w') as f:
        json.dump(r_config, f, indent=2)
    
    
    # Convert paths to absolute paths
    # paths = {
    #     'script': r_script_dir / "run_hmm.r",
    #     'util': r_script_dir / "util.r",  # Add util.r path
    #     'config': config_path.absolute(),
    #     'data': data_path.absolute(),
    #     'output': output_dir.absolute()
    # }
    
    # Build R command
    cmd = [
        str(config.r_executable or "Rscript"),
        str(r_script_path),
        str(util_path),
        str(config_path),
        str(target_data_path),
        str(dirs["base_dir"])
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        stream_r_output(process)

        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"R script failed with return code {process.returncode}")

        logging.info("HMM analysis completed successfully")
                
    except Exception as e:
        logging.error(f"Error running HMM analysis: {e}")
        raise

    except KeyboardInterrupt:
        logging.info("Received interrupt signal, terminating R process...")
        if process.poll() is None:  # If process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("R process didn't terminate, forcing kill...")
                process.kill()
        raise



def main():
    # Load configuration
    # config_path = Path("config/default.yaml")
    config_path = Path("config/RB_19_config.yaml")
    config = ConfigManager.load(config_path)
    
    # Set up logging with more detail
    logging.basicConfig(
        level=logging.DEBUG,  # Change to DEBUG for more detail
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting analysis...")
    
    # Create output directory
    output_dir = Path(config.analysis.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Run enabled analyses
    if "hmm" in config.analysis.enabled_analyses:
        logging.info("Running HMM analysis...")
        run_hmm_analysis(
            config=config.analysis,
            target_data_path=Path(config.analysis.target_dataset),
            output_dir=output_dir / "hmm"
        )
        

if __name__ == "__main__":
    main()