import os
import re
from datetime import datetime


def find_package_version(dir):
    package_versions = {}

    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if subdir_path.endswith(".egg-info") or subdir_path.endswith(".dist-info"):
            continue
        if not os.path.isdir(subdir_path):
            continue

        init_file_path = os.path.join(subdir_path, "__init__.py")

        with open(init_file_path, "r") as init_file:
            lines = list(line for line in init_file)
            # Search for the __version__ variable from the end of the file, to avoid taking the first value when __version__ is set multiple times, and instead get the final value.
            for line in reversed(lines):
                if "__version__" in line:
                    version_match = re.findall(r"^__version__\s*=\s*\"(\d+\.\d+(\.\d+)?)\"$", line)
                    if not version_match:
                        print(f"package={subdir_path} find __version__ but parse fail, line={line}")
                        continue
                    version = version_match[0][0]
                    package_versions[subdir] = version
                    break
            else:
                raise Exception(f"package={subdir_path} not find __version__")

    return package_versions


def add_metadata_info(package_dir, output_dir):
    package_versions = find_package_version(package_dir)

    os.makedirs(output_dir, exist_ok=True)

    for package_name, version in package_versions.items():
        dist_info_dir = os.path.join(output_dir, f"{package_name}-{version}.dist-info")
        os.makedirs(dist_info_dir, exist_ok=True)

        metadata_file = os.path.join(dist_info_dir, "METADATA")

        metadata_content = f"""Metadata-Version: {version}
Name: {package_name}
Version: {version}
Summary: A short description of my package.
Author: Your Name
Author-email: your.email@example.com
License: MIT
Description-Content-Type: text/plain
Description: A longer description of what this package does.
Home-page: https://github.com/yourusername/my_package"""

        with open(metadata_file, "w") as f:
            f.write(metadata_content.strip())

        print(f"Created METADATA for {package_name}-{version} at {metadata_file}")




