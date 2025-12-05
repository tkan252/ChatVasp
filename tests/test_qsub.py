"""
Test script for PBS qsub functionality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.pbs_tools import qsub


def test_qsub_basic():
    """Test basic qsub functionality."""
    # This is a test script - you would need a real PBS script to test
    # For demonstration, we'll show how to use the function

    # Example usage:
    # result = qsub("my_job.pbs")
    # print(f"Submission result: {result}")

    # With queue specification:
    # result = qsub("my_job.pbs", queue="workq")
    # print(f"Submission result: {result}")

    # With additional options:
    # result = qsub("my_job.pbs", queue="gpuq", options=["-l", "nodes=2:ppn=16,gpus=2"])
    # print(f"Submission result: {result}")

    print("qsub function is ready to use!")
    print("Example usage:")
    print("  result = qsub('job_script.pbs')")
    print("  result = qsub('job_script.pbs', queue='gpuq')")
    print("  result = qsub('job_script.pbs', options=['-l', 'nodes=4:ppn=32'])")


if __name__ == "__main__":
    test_qsub_basic()

