"""
PBS (Portable Batch System) management tools for querying and managing jobs.
Provides common PBS operations like qstat, qsub, qdel, qhold, qrls.
"""
import subprocess
from typing import Dict, List, Optional, Tuple, Union


def _run_pbs_command(command: List[str]) -> Tuple[str, int]:
    """
    Execute a PBS command and return output and exit code.
    
    Args:
        command: List of command and arguments
        
    Returns:
        Tuple of (stdout, exit_code)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out", 1
    except Exception as e:
        return f"Error executing command: {str(e)}", 1


def qstat_all() -> Dict[str, Union[str, List[Dict]]]:
    """
    Query all jobs in the queue (running and queued).
    Equivalent to: qstat -a
    
    Returns:
        Dictionary with 'status' and 'jobs' (list of job info dictionaries)
    """
    stdout, exit_code = _run_pbs_command(["qstat", "-a"])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "jobs": []
        }
    
    # Parse qstat output
    lines = stdout.strip().split('\n')
    jobs = []
    
    # Skip header lines (usually first 1-2 lines)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('-') and 'Job id' not in line.lower():
            start_idx = i
            break
    
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            job_id = parts[0]
            username = parts[1] if len(parts) > 1 else ""
            jobname = parts[2] if len(parts) > 2 else ""
            state = parts[4] if len(parts) > 4 else ""
            
            jobs.append({
                "job_id": job_id,
                "username": username,
                "jobname": jobname,
                "state": state,
                "raw": line
            })
    
    return {
        "status": "success",
        "jobs": jobs,
        "count": len(jobs)
    }


def qstat_user(username: Optional[str] = None) -> Dict[str, Union[str, List[Dict]]]:
    """
    Query jobs for a specific user.
    Equivalent to: qstat -u username
    
    Args:
        username: Username to query (if None, queries current user)
        
    Returns:
        Dictionary with 'status' and 'jobs' (list of job info dictionaries)
    """
    if username:
        stdout, exit_code = _run_pbs_command(["qstat", "-u", username])
    else:
        # Query current user's jobs
        stdout, exit_code = _run_pbs_command(["qstat"])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "jobs": []
        }
    
    # Parse similar to qstat_all
    lines = stdout.strip().split('\n')
    jobs = []
    
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('-') and 'Job id' not in line.lower():
            start_idx = i
            break
    
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            job_id = parts[0]
            jobname = parts[2] if len(parts) > 2 else ""
            state = parts[4] if len(parts) > 4 else ""
            
            jobs.append({
                "job_id": job_id,
                "jobname": jobname,
                "state": state,
                "raw": line
            })
    
    return {
        "status": "success",
        "jobs": jobs,
        "count": len(jobs)
    }


def qstat_job(job_id: str) -> Dict[str, Union[str, Dict]]:
    """
    Get detailed information about a specific job.
    Equivalent to: qstat -f job_id
    
    Args:
        job_id: PBS job ID (e.g., "12345.cluster" or "12345")
        
    Returns:
        Dictionary with job details
    """
    stdout, exit_code = _run_pbs_command(["qstat", "-f", job_id])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "job_id": job_id
        }
    
    # Parse qstat -f output (key-value pairs)
    details = {}
    current_key = None
    current_value = []
    
    for line in stdout.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts a new key (no leading whitespace or starts with Job Id)
        if ':' in line and not line.startswith(' ') and not line.startswith('\t'):
            # Save previous key-value
            if current_key:
                details[current_key] = ' '.join(current_value).strip()
            
            # Start new key-value
            parts = line.split(':', 1)
            current_key = parts[0].strip()
            current_value = [parts[1].strip()] if len(parts) > 1 else []
        else:
            # Continuation of previous value
            if current_key:
                current_value.append(line)
    
    # Save last key-value
    if current_key:
        details[current_key] = ' '.join(current_value).strip()
    
    return {
        "status": "success",
        "job_id": job_id,
        "details": details
    }


def qstat_finished() -> Dict[str, Union[str, List[Dict]]]:
    """
    Query finished jobs (completed or failed).
    Equivalent to: qstat -x
    
    Returns:
        Dictionary with 'status' and 'jobs' (list of finished job info)
    """
    stdout, exit_code = _run_pbs_command(["qstat", "-x"])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "jobs": []
        }
    
    # Parse similar to qstat_all
    lines = stdout.strip().split('\n')
    jobs = []
    
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('-') and 'Job id' not in line.lower():
            start_idx = i
            break
    
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            job_id = parts[0]
            username = parts[1] if len(parts) > 1 else ""
            jobname = parts[2] if len(parts) > 2 else ""
            state = parts[4] if len(parts) > 4 else ""
            
            jobs.append({
                "job_id": job_id,
                "username": username,
                "jobname": jobname,
                "state": state,
                "raw": line
            })
    
    return {
        "status": "success",
        "jobs": jobs,
        "count": len(jobs)
    }


def qsub(script_path: str, queue: Optional[str] = None, options: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Submit a PBS job script.
    Equivalent to: qsub [-q queue] [-l options...] script_path

    Args:
        script_path: Path to the PBS job script file
        queue: Optional queue name to submit to (e.g., "workq", "gpuq")
        options: Optional list of additional qsub options (e.g., ["-l", "nodes=2:ppn=16"])

    Returns:
        Dictionary with submission status and job ID if successful
    """
    command = ["qsub"]

    # Add queue option if specified
    if queue:
        command.extend(["-q", queue])

    # Add additional options if specified
    if options:
        command.extend(options)

    # Add the script path
    command.append(script_path)

    stdout, exit_code = _run_pbs_command(command)

    if exit_code == 0:
        # qsub typically returns the job ID on success
        job_id = stdout.strip()
        # Validate that we got a job ID (should be something like "12345.servername")
        if job_id and any(char.isdigit() for char in job_id):
            return {
                "status": "success",
                "job_id": job_id,
                "message": f"Job submitted successfully with ID: {job_id}"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to extract job ID from qsub output",
                "output": stdout
            }
    else:
        return {
            "status": "error",
            "message": f"qsub failed: {stdout.strip()}",
            "exit_code": exit_code
        }


def qdel(job_id: Union[str, List[str]]) -> Dict[str, str]:
    """
    Delete (cancel) one or more PBS jobs.
    Equivalent to: qdel job_id

    Args:
        job_id: PBS job ID(s) - can be a single string or list of strings

    Returns:
        Dictionary with operation status
    """
    if isinstance(job_id, str):
        job_ids = [job_id]
    else:
        job_ids = job_id
    
    results = []
    for jid in job_ids:
        stdout, exit_code = _run_pbs_command(["qdel", jid])
        results.append({
            "job_id": jid,
            "status": "success" if exit_code == 0 else "error",
            "message": stdout.strip()
        })
    
    return {
        "status": "success" if all(r["status"] == "success" for r in results) else "partial",
        "results": results
    }


def qhold(job_id: Union[str, List[str]]) -> Dict[str, str]:
    """
    Hold (suspend) one or more PBS jobs.
    Equivalent to: qhold job_id
    
    Args:
        job_id: PBS job ID(s) - can be a single string or list of strings
        
    Returns:
        Dictionary with operation status
    """
    if isinstance(job_id, str):
        job_ids = [job_id]
    else:
        job_ids = job_id
    
    results = []
    for jid in job_ids:
        stdout, exit_code = _run_pbs_command(["qhold", jid])
        results.append({
            "job_id": jid,
            "status": "success" if exit_code == 0 else "error",
            "message": stdout.strip()
        })
    
    return {
        "status": "success" if all(r["status"] == "success" for r in results) else "partial",
        "results": results
    }


def qrls(job_id: Union[str, List[str]]) -> Dict[str, str]:
    """
    Release (unhold) one or more PBS jobs.
    Equivalent to: qrls job_id
    
    Args:
        job_id: PBS job ID(s) - can be a single string or list of strings
        
    Returns:
        Dictionary with operation status
    """
    if isinstance(job_id, str):
        job_ids = [job_id]
    else:
        job_ids = job_id
    
    results = []
    for jid in job_ids:
        stdout, exit_code = _run_pbs_command(["qrls", jid])
        results.append({
            "job_id": jid,
            "status": "success" if exit_code == 0 else "error",
            "message": stdout.strip()
        })
    
    return {
        "status": "success" if all(r["status"] == "success" for r in results) else "partial",
        "results": results
    }


def qstat_queues() -> Dict[str, Union[str, List[Dict]]]:
    """
    Query queue information.
    Equivalent to: qstat -q
    
    Returns:
        Dictionary with 'status' and 'queues' (list of queue info dictionaries)
        Each queue dictionary contains queue_name and other queue attributes
    """
    stdout, exit_code = _run_pbs_command(["qstat", "-q"])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "queues": []
        }
    
    # Parse qstat -q output
    # Typical format:
    # Queue            Max   Tot   Ena   Str   Que   Run   Hld   Wat   Trn   Ext Type
    # --------------------------------------------------------------------------------
    # batch             -     -     -     -     -     -     -     -     -     -     E
    # or with detailed info per queue
    
    lines = stdout.strip().split('\n')
    queues = []
    
    # Find header line and data start
    header_found = False
    start_idx = 0
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Look for header line (contains "Queue" or column names)
        if 'queue' in line_lower and ('max' in line_lower or 'tot' in line_lower or 'type' in line_lower):
            header_found = True
            start_idx = i + 1
            # Skip separator line (dashes)
            if start_idx < len(lines) and lines[start_idx].strip().startswith('-'):
                start_idx += 1
            break
    
    # If no header found, try to find first data line
    if not header_found:
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('-') and 'Queue' not in line:
                start_idx = i
                break
    
    # Parse queue information
    for line in lines[start_idx:]:
        line = line.strip()
        if not line or line.startswith('-'):
            continue
        
        parts = line.split()
        if not parts:
            continue
        
        # First part is usually queue name
        queue_name = parts[0]
        queue_info = {"queue_name": queue_name}
        
        # Try to parse common columns (if header format is standard)
        # Format: queue_name max tot ena str que run hld wat trn ext type
        if len(parts) >= 2:
            # Common fields (may vary by PBS version)
            field_names = ["max", "tot", "ena", "str", "que", "run", "hld", "wat", "trn", "ext", "type"]
            for idx, field in enumerate(field_names, start=1):
                if idx < len(parts):
                    queue_info[field] = parts[idx]
        
        # Also store raw line for reference
        queue_info["raw"] = line
        
        queues.append(queue_info)
    
    return {
        "status": "success",
        "queues": queues,
        "count": len(queues)
    }


def get_job_status_summary() -> Dict[str, Union[int, str]]:
    """
    Get a summary of job statuses (running, queued, held, etc.).
    
    Returns:
        Dictionary with counts of jobs in different states
    """
    result = qstat_all()
    
    if result["status"] != "success":
        return {
            "status": "error",
            "message": result.get("message", "Failed to query jobs")
        }
    
    summary = {
        "status": "success",
        "total": len(result["jobs"]),
        "running": 0,
        "queued": 0,
        "held": 0,
        "other": 0
    }
    
    for job in result["jobs"]:
        state = job.get("state", "").upper()
        if "R" in state:  # Running
            summary["running"] += 1
        elif "Q" in state:  # Queued
            summary["queued"] += 1
        elif "H" in state:  # Held
            summary["held"] += 1
        else:
            summary["other"] += 1
    
    return summary


__all__ = [
    "qstat_all",
    "qstat_user",
    "qstat_job",
    "qstat_finished",
    "qstat_queues",
    "qsub",
    "qdel",
    "qhold",
    "qrls",
    "get_job_status_summary",
]

