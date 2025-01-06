#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def create_virtual_environment(venv_path: Path):
    """Creates a virtual environment."""
    print(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def install_dependencies(venv_path: Path, requirements_file: Path):
    """Installs project dependencies."""
    print(f"Installing dependencies from {requirements_file}...")
    pip_executable = venv_path / "bin" / "pip"
    try:
        subprocess.run([str(pip_executable), "install", "-r", str(requirements_file)], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def install_dev_tools(venv_path: Path):
    """Installs development tools."""
    print("Installing development tools...")
    pip_executable = venv_path / "bin" / "pip"
    dev_tools = [
        "black",
        "isort",
        "flake8",
        "mypy",
        "pytest",
        "pytest-cov",
        "pre-commit"
    ]
    try:
        subprocess.run([str(pip_executable), "install"] + dev_tools, check=True)
        print("Development tools installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing development tools: {e}")
        sys.exit(1)

def setup_pre_commit_hooks(venv_path: Path):
    """Sets up pre-commit hooks."""
    print("Setting up pre-commit hooks...")
    pre_commit_executable = venv_path / "bin" / "pre-commit"
    try:
        subprocess.run([str(pre_commit_executable), "install"], check=True)
        print("Pre-commit hooks set up successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up pre-commit hooks: {e}")
        sys.exit(1)

def download_data(data_url: str, data_dir: Path):
    """Downloads necessary data."""
    print(f"Downloading data from {data_url} to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["wget", "-P", str(data_dir), data_url], check=True)
        print("Data downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

def configure_ide_settings(project_root: Path):
    """Configures IDE settings."""
    print("Configuring IDE settings...")
    vscode_dir = project_root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "python.defaultInterpreterPath": str(project_root / "venv" / "bin" / "python"),
        "python.formatting.provider": "black",
        "python.linting.flake8Enabled": True,
        "python.linting.mypyEnabled": True,
        "python.sortImports.args": ["--profile", "black"],
        "[python]": {
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            }
        }
    }
    with open(vscode_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    print("IDE settings configured successfully.")

def run_tests(venv_path: Path):
    """Runs initial tests."""
    print("Running initial tests...")
    pytest_executable = venv_path / "bin" / "pytest"
    try:
        subprocess.run([str(pytest_executable), "."], check=True)
        print("Tests passed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

def print_setup_instructions(venv_path: Path):
    """Prints setup instructions."""
    print("\nDevelopment environment setup completed successfully!")
    print("\nTo activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    print("\nTo run the tests, use:")
    print("  pytest")
    print("\nTo format the code, use:")
    print("  black .")
    print("  isort .")
    print("\nHappy coding!")

def main():
    """Sets up the development environment."""
    project_root = Path(__file__).parent
    venv_path = project_root / "venv"
    requirements_file = project_root / "requirements.txt"
    data_dir = project_root / "data"

    create_virtual_environment(venv_path)
    install_dependencies(venv_path, requirements_file)
    install_dev_tools(venv_path)
    setup_pre_commit_hooks(venv_path)
    # download_data("your_data_url", data_dir)  # Uncomment and replace with actual URL if needed
    configure_ide_settings(project_root)
    run_tests(venv_path)
    print_setup_instructions(venv_path)

if __name__ == "__main__":
    main()
