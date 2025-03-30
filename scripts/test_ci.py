#!/usr/bin/env python
"""
Local CI script to run tests and linting checks.
This script simulates what the CI/CD pipeline would do.
"""

import os
import subprocess
import sys


def run_command(command):
    """Run a command and return the result."""
    print(f"\n\033[1;34mRunning: {command}\033[0m")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"\033[1;31mError: {result.stderr}\033[0m")
    return result.returncode


def main():
    """Run the CI checks."""
    print("\033[1;32m==== Starting Local CI Checks ====\033[0m")
    
    # Test if required tools are installed
    if run_command("pip show pytest") != 0:
        print("\033[1;31mPytest is not installed. Installing...\033[0m")
        run_command("pip install pytest")
    
    if run_command("pip show flake8") != 0:
        print("\033[1;31mFlake8 is not installed. Installing...\033[0m")
        run_command("pip install flake8")
    
    # Run linting but only on our test files for now
    print("\n\033[1;32m==== Running Linting Checks ====\033[0m")
    lint_result = run_command("flake8 src/tests/ --max-line-length=120")
    
    # Run tests
    print("\n\033[1;32m==== Running Tests ====\033[0m")
    test_result = run_command("python -m pytest src/tests/ -v")
    
    # Summary
    if lint_result == 0 and test_result == 0:
        print("\n\033[1;32m==== All CI Checks Passed! ====\033[0m")
        return 0
    else:
        print("\n\033[1;31m==== CI Checks Failed! ====\033[0m")
        if lint_result != 0:
            print("\033[1;31m- Linting checks failed\033[0m")
        if test_result != 0:
            print("\033[1;31m- Tests failed\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 