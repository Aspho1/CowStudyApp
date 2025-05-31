# test_visuals_config.py (create this file in your project root)
from pathlib import Path
from cowstudyapp.config import VisualsConfig

# Print module path to confirm we're importing the right one
import cowstudyapp.config
print(f"Using config module from: {cowstudyapp.config.__file__}")

# Directly try to instantiate a VisualsConfig with minimal args
try:
    config = VisualsConfig(
        visuals_root_path=Path("/tmp/test"),
        predictions_path=Path("/tmp/test2")
    )
    print("SUCCESS! VisualsConfig created with defaults.")
    print(f"Radar config: {config.radar}")
    print(f"Domain config: {config.domain}")
except Exception as e:
    print(f"FAILED to create VisualsConfig: {e}")

# Also print the class definition to verify default_factory settings







