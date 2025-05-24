# src/run_analysis.py
from datetime import datetime
from cowstudyapp.config import (
    ConfigManager,
    AnalysisConfig,
    HMMConfig,
    LSTMConfig,
    DistributionTypes,
    AnalysisModes
)
from pathlib import Path
import subprocess
import logging
import json
import platform
import os
import sys
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def find_r_executable(config_path):
    """Find the R executable on the system"""
    if platform.system() == "Windows":
        # Common R installation paths on Windows
        possible_paths = [
            config_path, 
            r"C:/Program Files/R/R-4.4.3/bin/Rscript.exe",
            r"C:/Program Files/R/R-4.4.0/bin/Rscript.exe",
            r"C:/Program Files/R/R-4.3.2/bin/Rscript.exe",
            r"C:/Program Files/R/R-4.3.0/bin/Rscript.exe",
        ]

        # Check R_HOME environment variable
        r_home = os.environ.get("R_HOME")
        print("RHOME:", r_home)
        if r_home:
            possible_paths.append(Path(r_home) / "bin" / "Rscript.exe")

        # # Try to find Rscript in PATH
        # try:
        #     result = subprocess.run(
        #         ["where", "Rscript.exe"], capture_output=True, text=True
        #     )
        #     if result.returncode == 0:
        #         possible_paths.extend(result.stdout.splitlines())
        # except subprocess.SubprocessError:
        #     pass



        # Return first existing path
        for path in possible_paths:
            print("CHECKING ", path)
            if Path(path).exists():
                return str(Path(path))

    else:  # Linux/Mac
        try:
            return subprocess.check_output(["which", "Rscript"]).decode().strip()
        except subprocess.SubprocessError:
            pass

    raise FileNotFoundError(
        "Could not find R executable. Please install R and ensure it's in your PATH or"
        "designated in the `r_executable` field in your configuration file."
    )


def install_r_packages(packages):
    """Install missing R packages"""
    quoted_packages = ", ".join(f"'{pkg.strip()}'" for pkg in packages)
    r_script = f"""
    install.packages(c({quoted_packages}), 
                    repos='https://cran.rstudio.com/')
    """

    r_executable = find_r_executable()
    try:
        subprocess.run([r_executable, "--vanilla", "-e", r_script], check=True)
        logger.info(f"Successfully installed R packages: {packages}")
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to install R packages: {e}")

def check_r_packages(r_executable):
    """Check if required R packages are installed"""
    # List of required packages
    required_packages = [
        "momentuHMM",
        "dplyr",
        "ggplot2",
        "caret",
        "fitdistrplus",
        "circular",
        "CircStats",
        "lubridate",
        "grid",
        "gridExtra",
        "movMF",
        "suncalc",
        "lme4",
        "fs"
    ]


    # Step 1: Just get installed packages
    try:
        # First just try to get the list of installed packages
        list_cmd = subprocess.run(
            [r_executable, '--vanilla',"-e", "cat(paste(installed.packages()[,1], collapse=','))"],
            capture_output=True,
            text=True,
            check=True,
        )

        installed_packages = list_cmd.stdout.split(",")
        logger.info(f"Found {len(installed_packages)} installed R packages")

        # Step 2: Check which required packages are missing
        missing_packages = [
            pkg for pkg in required_packages if pkg not in installed_packages
        ]

        if missing_packages:
            # Ask user if they want to install missing packages
            print(f"\nMissing R packages: {', '.join(missing_packages)}")
            # response = input("Would you like to install them? [y/N] ")
            response = 'y'

            if response.lower() == "y":
                # Install packages one at a time
                for pkg in missing_packages:
                    print(f"\nInstalling {pkg}...")
                    install_cmd = [
                        r_executable,
                        '--vanilla',
                        "-e",
                        f"install.packages('{pkg}', repos='https://cran.rstudio.com/', dependencies=TRUE)",
                    ]
                    try:
                        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                        print("=== R stdout ===\n", result.stdout)
                        print("=== R stderr ===\n", result.stderr)
                        result.check_returncode()                        
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
            logger.info("All required R packages are installed")

    except subprocess.CalledProcessError as e:
        # Add more diagnostic information
        error_msg = (
            f"Error running R command:\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"STDOUT: {e.stdout if e.stdout else 'Empty'}\n"
            f"STDERR: {e.stderr if e.stderr else 'Empty'}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def stream_r_output(process, prefix="R"):
    """Stream output from an R process in real-time with interrupt handling"""
    try:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line:
                logger.info(f"{prefix}: {line}")
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
        DistributionTypes.WRAPPEDCAUCHY: "wrpcauchy",
    }
    return mapping[dist_type]


def create_analysis_directories_OLD(
    base_dir: Path, features: list, analysis_type: str, dataset_name: str, subdirs: bool = True
) -> dict:
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

    # # Create feature identifier
    # feature_names = sorted([f.name for f in features])
    # feature_id = "-".join(feature_names)
    # if len(feature_id) > 50:  # Limit length
    #     feature_id = feature_id[:50]

    # Create directory structure
    analysis_dir = base_dir / analysis_type / (dataset_name + "_" + timestamp)

    # if subdirs == True:
    # Create subdirectories
    plots_dir = analysis_dir / "plots"
    dist_plots_dir = plots_dir / "distributions"
    # models_dir = analysis_dir / "models"  # New directory for saved models

    # Create all directories
    for directory in [analysis_dir, plots_dir, dist_plots_dir]:  # , models_dir
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": analysis_dir,
        "plots_dir": plots_dir,
        "dist_plots_dir": dist_plots_dir,
        # "models_dir": models_dir
    }


def create_analysis_directories(cv_dir: Path, mod_dir: Path, pred_dir: Path) -> dict:
    """
    Create directory structure for HMM analysis outputs

    Args:
        base_dir: Base output directory (data/analysis_results/hmm)
        features: List of features being used
        analysis_type: Either 'LOOCV' or 'Product'

    Returns:
        Dictionary containing paths to different output directories
    """

    cv_dir = cv_dir / "HMM"
    mod_dir = mod_dir / "HMM"
    pred_dir = pred_dir / "HMM"

    # models_dir = analysis_dir / "models"  # New directory for saved models
    # Create all directories
    for directory in [cv_dir, mod_dir, pred_dir]:  # , models_dir
        deepest_dir: Path = (directory / "plots" / "distributions")
        deepest_dir.mkdir(parents=True, exist_ok=True)


    return {
        "cv_dir": cv_dir,
        "mod_dir": mod_dir,
        "pred_dir": pred_dir,
        # "models_dir": models_dir
    }


# def run_hmm_analysis(config: ConfigManager, target_data_path: Path, output_dir: Path):
def run_hmm_analysis(config: ConfigManager, target_data_path: Path, progress_callback=None):
    """Run the HMM analysis using R scripts"""

    if progress_callback is None:
        progress_callback = lambda percent, message: print(f"{percent}%: {message}")

    progress_callback(7, "Locating R executable...")
    analysis = config.analysis
    r_executable = find_r_executable(analysis.r_executable)
    print(r_executable)
    check_r_packages(r_executable)

    # Ensure R script directory exists
    ########## This needs cleaning
    try:
        r_script_dir = Path("src/cowstudyapp/analysis/HMM")
    except FileNotFoundError:
        r_script_dir = Path("/SShare/Education/CowStudyApp/src/cowstudyapp/analysis/HMM")
    
    if not r_script_dir.exists():
        raise FileNotFoundError(f"R script directory not found: {r_script_dir}")

    # analysis_type = config.mode if hasattr(config, 'mode') else "LOOCV"
    if analysis.hmm is None:
        raise ValueError("ERROR: Missing required HMM section in analysis config")  
    
    progress_callback(8, "Creating directories...")
    dirs = create_analysis_directories(
        cv_dir=config.analysis.cv_results,
        mod_dir=config.analysis.models,
        pred_dir=config.analysis.predictions,
    )

    # if analysis.mode == "LOOCV":
    #     curdir = config.analysis.cv_results
    # else:
    #     if not analysis.training_info:
    #         curdir = config.analysis.models
    #     else:
    #         curdir = config.analysis.predictions


    r_script_path = r_script_dir / "run_hmm.r"
    util_path = r_script_dir / "util.r"
    # Create output directory
    # output_dir.mkdir(parents=True, exist_ok=True)

    training_info_type = (
        "dataset"
        if not analysis.training_info
        else analysis.training_info.training_info_type
    )
    training_info_path = (
        str(target_data_path)
        if not analysis.training_info
        else analysis.training_info.training_info_path
    )

    progress_callback(9, "Setting Config...")
    # Convert Python config to R format
    r_config = {
        "states": analysis.hmm.states,
        "random_seed": analysis.random_seed,
        "all_states": config.labels.valid_activities,
        "features": [feature.model_dump() for feature in analysis.hmm.features],
        "mode": analysis.mode,
        "time_covariate": analysis.hmm.time_covariate,
        "timezone": analysis.timezone,
        "excluded_devices": analysis.excluded_devices,
        "day_only": analysis.day_only,
        "options": {
            **analysis.hmm.options,
            "distributions": {
                "regular": [
                    get_r_dist_name(DistributionTypes(d))
                    for d in analysis.hmm.options["distributions"]["regular"]
                ],
                "circular": [
                    get_r_dist_name(DistributionTypes(d))
                    for d in analysis.hmm.options["distributions"]["circular"]
                ],
            },
        },
    }

    if analysis.training_info is not None:
        r_config["training_info"] = (
            {
                "training_info_type": training_info_type,
                "training_info_path": training_info_path,
            },
        )

    config_path: Path = dirs['mod_dir'] / "hmm_config.json"
    with open(config_path, "w") as f:
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
        str(r_executable or "Rscript"),
        str(r_script_path),
        str(util_path),
        str(config_path),
        str(target_data_path),
        str(dirs["cv_dir"]),
        str(dirs["mod_dir"]),
        str(dirs["pred_dir"])
    ]

    progress_callback(10, "Starting R subprocess... This may take a while.")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        stream_r_output(process)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"R script failed with return code {process.returncode}")

        logger.info("HMM analysis completed successfully")

    except Exception as e:
        logger.error(f"Error running HMM analysis: {e}")
        raise

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, terminating R process...")
        if process.poll() is None:  # If process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("R process didn't terminate, forcing kill...")
                process.kill()
        raise


def run_lstm_analysis(config: ConfigManager, target_data_path: Path, progress_callback=None):
    """Run the HMM analysis using R scripts"""
    from cowstudyapp.analysis.RNN.run_lstm import LSTM_Model

    analysis = config.analysis
    try:
        if analysis.lstm is None:
            raise ValueError("ERROR: Missing required LSTM section in analysis config")  

        lstm = LSTM_Model(config=config)

        lstm.run_LSTM(progress_callback=progress_callback)

        logger.info("LSTM analysis completed successfully")

    except Exception as e:
        logger.error(f"Error running LSTM analysis: {e}")
        raise




def main(config_path=None, progress_callback=None):
    if progress_callback is None:
        progress_callback = lambda percent, message: print(f"{percent}%: {message}")
    logger.info("Starting run_analysis")
    if config_path is None and len(sys.argv) > 1:
        print('here')
        parser = argparse.ArgumentParser(
            prog="run_analysis",
            description="Perform serial classification on data"
        )
        parser.add_argument(
            "task_id",
            help="ID of this analysis task (for logging, etc.)"
        )
        parser.add_argument(
            "config_path",
            type=Path,
            help="YAML file with analysis configuration"
        )
        args = parser.parse_args()
        config_path = args.config_path
        task_id = args.task_id

    else:
        # Load configuration
        # config_path = Path("config/default.yaml")
        if not config_path:
            config_path = Path("config/RB_22_config.yaml")
        config_path = config_path
        # config_path = Path("config/RB_19_22_combined_config.yaml")
        task_id = "direct-call"
        
    config = ConfigManager.load(config_path)
    progress_callback(6, f"Config {config_path} loaded")
    try:
        if config.analysis.hmm.enabled:
            logger.info("Running HMM analysis...")
            run_hmm_analysis(
                config=config,
                target_data_path=Path(config.analysis.target_dataset),
                progress_callback=progress_callback
            )
        if config.analysis.lstm.enabled:
            logger.info("Running LSTM analysis...")
            run_lstm_analysis(
                config=config,
                target_data_path=Path(config.analysis.target_dataset),
                progress_callback=progress_callback
            )

        return True

    except Exception as e:
        logger.exception("Error while performing task %s: %s", task_id, str(e))
        return False


# & C:/Users/thl31/pythonVenvs/CowStudyApp/Scripts/python.exe o:/Education/CowStudyApp/src/cowstudyapp/run_analysis.py "direct-call" "config/RB_22_config.yaml"

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

