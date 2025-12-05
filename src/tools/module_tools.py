"""
Environment Modules (module avail) management tools.
Provides functions to query available software modules.
"""
import subprocess
import re
from typing import Dict, List, Optional, Tuple


def _run_module_command(command: List[str]) -> Tuple[str, int]:
    """
    Execute a module command and return output and exit code.
    
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
            timeout=60,  # module avail can be slow
            shell=False
        )
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out", 1
    except Exception as e:
        return f"Error executing command: {str(e)}", 1


def module_avail(pattern: Optional[str] = None) -> Dict[str, any]:
    """
    Get list of available software modules.
    Equivalent to: module avail [pattern]
    
    Args:
        pattern: Optional pattern to filter modules (e.g., "vasp", "intel")
        
    Returns:
        Dictionary with 'status', 'modules' (list of module names), and 'count'
    """
    if pattern:
        stdout, exit_code = _run_module_command(["module", "avail", pattern])
    else:
        stdout, exit_code = _run_module_command(["module", "avail"])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "modules": [],
            "count": 0
        }
    
    # Parse module avail output
    # The output format varies by Environment Modules version:
    # - Old format: Lists modules with paths, separators
    # - New format (2.0+): More structured output
    
    modules = []
    lines = stdout.strip().split('\n')
    
    # Common words to filter out
    skip_words = {
        'module', 'modules', 'path', 'paths', 'available', 'where', 'use',
        'currently', 'loaded', 'default', 'version', 'versions'
    }
    
    # Pattern to match module names (name/version or just name)
    # Module names can contain: letters, numbers, dots, dashes, underscores, slashes
    module_pattern = re.compile(r'^[\w][\w/.-]*[\w]$|^[\w]+$')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip separator lines (lines with only dashes, equals, or other separators)
        if re.match(r'^[-=~_]+$', line):
            continue
        
        # Skip header lines
        if any(keyword in line.lower() for keyword in ['currently loaded', 'where:', 'use module']):
            continue
        
        # Skip lines that are clearly not module names (too long, contain special chars)
        if len(line) > 100 or ':' in line and not '/' in line:
            # Might be a path description, skip
            continue
        
        # Try to extract module names from the line
        # Modules might be space-separated in columns
        parts = line.split()
        
        for part in parts:
            # Clean up: remove trailing colons, parentheses, etc.
            part = part.rstrip(':,;()[]').strip()
            
            # Check if it matches module name pattern
            if module_pattern.match(part):
                part_lower = part.lower()
                
                # Filter out common non-module words
                if part_lower not in skip_words and len(part) > 0:
                    # Additional validation: module names usually have at least one letter
                    if re.search(r'[a-zA-Z]', part):
                        if part not in modules:
                            modules.append(part)
    
    # If still no modules found, try a more aggressive regex approach
    if len(modules) == 0:
        # Look for patterns like "name/version" throughout the output
        # This pattern matches: word characters, optionally followed by / and version
        potential_modules = re.findall(r'\b([a-zA-Z][\w.-]*(?:/[\w.]+)?)\b', stdout)
        for mod in potential_modules:
            mod_lower = mod.lower()
            # Filter out common non-module words and very short strings
            if (mod_lower not in skip_words and 
                len(mod) >= 2 and 
                re.search(r'[a-zA-Z]', mod)):
                if mod not in modules:
                    modules.append(mod)
    
    # Sort modules for easier reading
    modules.sort()
    
    return {
        "status": "success",
        "modules": modules,
        "count": len(modules),
        "pattern": pattern if pattern else None
    }


def module_avail_search(search_term: str) -> Dict[str, any]:
    """
    Search for available modules matching a search term.
    This is a convenience wrapper around module_avail with pattern matching.
    
    Args:
        search_term: Term to search for in module names (e.g., "vasp", "intel", "gcc")
        
    Returns:
        Dictionary with 'status', 'modules' (filtered list), and 'count'
    """
    result = module_avail()
    
    if result["status"] != "success":
        return result
    
    # Filter modules that contain the search term (case-insensitive)
    search_lower = search_term.lower()
    filtered_modules = [
        mod for mod in result["modules"]
        if search_lower in mod.lower()
    ]
    
    return {
        "status": "success",
        "modules": filtered_modules,
        "count": len(filtered_modules),
        "search_term": search_term
    }


def module_list() -> Dict[str, any]:
    """
    Get list of currently loaded modules.
    Equivalent to: module list
    
    Returns:
        Dictionary with 'status', 'modules' (list of loaded module names), and 'count'
    """
    stdout, exit_code = _run_module_command(["module", "list"])
    
    if exit_code != 0:
        return {
            "status": "error",
            "message": stdout,
            "modules": [],
            "count": 0
        }
    
    # Parse module list output
    # Format is typically: "Currently Loaded Modules:"
    # followed by module names, one per line or space-separated
    
    modules = []
    lines = stdout.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or 'currently loaded' in line.lower() or line.startswith('-'):
            continue
        
        # Extract module names (format: name/version or just name)
        parts = line.split()
        for part in parts:
            part = part.strip()
            if re.match(r'^[\w/.-]+$', part) and len(part) > 0:
                if part.lower() not in ['module', 'modules', 'loaded']:
                    if part not in modules:
                        modules.append(part)
    
    return {
        "status": "success",
        "modules": modules,
        "count": len(modules)
    }


__all__ = [
    "module_avail",
    "module_avail_search",
    "module_list",
]

