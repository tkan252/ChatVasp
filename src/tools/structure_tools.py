"""
Structure manipulation tools using pymatgen.

This module provides tools for common structure operations:
- Supercell generation
- Surface/slab creation
- Adsorbate placement
- Element substitution

All tools are wrapped in a StaticWorkbench for use with AutoGen agents.
"""

from typing import Annotated, Literal, Optional
from autogen_core.tools import FunctionTool, StaticWorkbench


def make_supercell(
    input_path: Annotated[
        str,
        "Path to the input structure file (POSCAR, CIF, etc.)."
    ],
    output_path: Annotated[
        str,
        "Path where the supercell structure will be saved."
    ],
    scaling_matrix: Annotated[
        str,
        "Scaling matrix as 'a,b,c' for diagonal (e.g., '2,2,1') or "
        "'a11,a12,a13,a21,a22,a23,a31,a32,a33' for full 3x3 matrix."
    ]
) -> str:
    """
    Create a supercell from the input structure.
    
    This tool expands the unit cell by the specified scaling matrix.
    Use diagonal scaling (e.g., '2,2,1') for simple expansions or
    a full 3x3 matrix for more complex transformations.
    
    Args:
        input_path: Path to input structure file.
        output_path: Path for output supercell structure.
        scaling_matrix: Scaling factors as comma-separated string.
    
    Returns:
        str: Success message with supercell details.
    
    Example:
        make_supercell("POSCAR", "POSCAR_2x2x1", "2,2,1")
    """
    from pymatgen.core import Structure
    
    structure = Structure.from_file(input_path)
    original_atoms = len(structure)
    
    # Parse scaling matrix
    values = [int(x.strip()) for x in scaling_matrix.split(',')]
    if len(values) == 3:
        # Diagonal scaling
        matrix = [[values[0], 0, 0], [0, values[1], 0], [0, 0, values[2]]]
    elif len(values) == 9:
        # Full 3x3 matrix
        matrix = [values[0:3], values[3:6], values[6:9]]
    else:
        return f"Error: Invalid scaling matrix. Use 'a,b,c' or 9 values for 3x3 matrix."
    
    supercell = structure.make_supercell(matrix, in_place=False)
    supercell.to(filename=output_path)
    
    return (f"Supercell created at {output_path}\n"
            f"  Original: {original_atoms} atoms\n"
            f"  Supercell: {len(supercell)} atoms\n"
            f"  Scaling: {values[0] if len(values)==3 else 'custom'}x"
            f"{values[1] if len(values)==3 else ''}x{values[2] if len(values)==3 else ''}")


def make_slab(
    input_path: Annotated[
        str,
        "Path to the input bulk structure file (POSCAR, CIF, etc.)."
    ],
    output_path: Annotated[
        str,
        "Path where the slab structure will be saved."
    ],
    miller_index: Annotated[
        str,
        "Miller indices as 'h,k,l' (e.g., '1,1,1' for (111) surface)."
    ],
    min_slab_size: Annotated[
        float,
        "Minimum slab thickness in Angstroms."
    ] = 10.0,
    min_vacuum_size: Annotated[
        float,
        "Minimum vacuum thickness in Angstroms."
    ] = 15.0,
    center_slab: Annotated[
        bool,
        "Whether to center the slab in the cell."
    ] = True
) -> str:
    """
    Create a surface slab from a bulk structure.
    
    This tool generates a slab model with specified Miller indices,
    slab thickness, and vacuum layer. Useful for surface calculations.
    
    Args:
        input_path: Path to input bulk structure file.
        output_path: Path for output slab structure.
        miller_index: Miller indices as 'h,k,l'.
        min_slab_size: Minimum slab thickness in Angstroms.
        min_vacuum_size: Minimum vacuum thickness in Angstroms.
        center_slab: Whether to center the slab.
    
    Returns:
        str: Success message with slab details.
    
    Example:
        make_slab("POSCAR_bulk", "POSCAR_slab_111", "1,1,1", 10.0, 15.0)
    """
    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator
    
    structure = Structure.from_file(input_path)
    
    # Parse Miller indices
    hkl = tuple(int(x.strip()) for x in miller_index.split(','))
    if len(hkl) != 3:
        return "Error: Miller index must be 'h,k,l' format (e.g., '1,1,1')."
    
    # Generate slab
    slab_gen = SlabGenerator(
        structure,
        miller_index=hkl,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
        in_unit_planes=False
    )
    
    slabs = slab_gen.get_slabs()
    if not slabs:
        return f"Error: Could not generate slab for ({hkl[0]}{hkl[1]}{hkl[2]}) surface."
    
    # Take the first (most stable) slab
    slab = slabs[0]
    slab.to(filename=output_path)
    
    return (f"Slab created at {output_path}\n"
            f"  Surface: ({hkl[0]}{hkl[1]}{hkl[2]})\n"
            f"  Atoms: {len(slab)}\n"
            f"  Slab thickness: ~{min_slab_size} Å\n"
            f"  Vacuum: ~{min_vacuum_size} Å\n"
            f"  Total slabs generated: {len(slabs)} (saved the first one)")


def add_adsorbate(
    slab_path: Annotated[
        str,
        "Path to the input slab structure file."
    ],
    output_path: Annotated[
        str,
        "Path where the structure with adsorbate will be saved."
    ],
    adsorbate: Annotated[
        str,
        "Adsorbate species: single element (e.g., 'H', 'O') or "
        "molecule formula (e.g., 'H2O', 'CO', 'OH')."
    ],
    site_type: Annotated[
        Literal["ontop", "bridge", "hollow", "all"],
        "Adsorption site type. Use 'all' to get all possible sites."
    ] = "ontop",
    height: Annotated[
        float,
        "Height of adsorbate above the surface in Angstroms."
    ] = 2.5,
    repeat: Annotated[
        str,
        "Repeat pattern for adsorbate placement as 'a,b,c' (e.g., '1,1,1' for single adsorbate, "
        "'2,2,1' to place adsorbates on a 2x2 supercell pattern)."
    ] = "2,2,1"
) -> str:
    """
    Add an adsorbate molecule to a slab surface.
    
    This tool places an adsorbate at ALL found adsorption sites of the specified
    type on the surface. Supports common molecules and single atoms. The repeat 
    parameter controls how many copies of the adsorbate are placed.
    
    Output files will be suffixed with the site index (e.g. output_path_0, output_path_1).
    
    Args:
        slab_path: Path to input slab structure.
        output_path: Base path for output structures.
        adsorbate: Adsorbate species or molecule.
        site_type: Type of adsorption site.
        height: Height above surface in Angstroms.
        repeat: Repeat pattern for adsorbate placement (e.g., '2,2,1').
    
    Returns:
        str: Success message with details of generated files.
    
    Example:
        add_adsorbate("POSCAR_slab", "POSCAR_slab_H", "H", "ontop", 1.5, "2,2,1")
    """
    from pymatgen.core import Structure, Molecule
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    import os
    
    slab = Structure.from_file(slab_path)
    
    # Create adsorbate molecule
    # Common molecules with their geometries
    molecules = {
        'H': Molecule(['H'], [[0, 0, 0]]),
        'O': Molecule(['O'], [[0, 0, 0]]),
        'N': Molecule(['N'], [[0, 0, 0]]),
        'C': Molecule(['C'], [[0, 0, 0]]),
        'H2': Molecule(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]]),
        'O2': Molecule(['O', 'O'], [[0, 0, 0], [0, 0, 1.21]]),
        'N2': Molecule(['N', 'N'], [[0, 0, 0], [0, 0, 1.10]]),
        'CO': Molecule(['C', 'O'], [[0, 0, 0], [0, 0, 1.13]]),
        'CO2': Molecule(['C', 'O', 'O'], [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]),
        'H2O': Molecule(['O', 'H', 'H'], [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]),
        'OH': Molecule(['O', 'H'], [[0, 0, 0], [0, 0, 0.97]]),
        'NH3': Molecule(['N', 'H', 'H', 'H'], 
                       [[0, 0, 0], [0, 0.94, 0.38], [0.81, -0.47, 0.38], [-0.81, -0.47, 0.38]]),
        'CH4': Molecule(['C', 'H', 'H', 'H', 'H'],
                       [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], 
                        [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]),
    }
    
    if adsorbate.upper() in molecules:
        mol = molecules[adsorbate.upper()]
    elif adsorbate in molecules:
        mol = molecules[adsorbate]
    else:
        # Try single element
        try:
            mol = Molecule([adsorbate], [[0, 0, 0]])
        except Exception as e:
            return f"Error: Unknown adsorbate '{adsorbate}'. Supported: {list(molecules.keys())}"
    
    # Find adsorption sites
    asf = AdsorbateSiteFinder(slab)
    sites_dict = asf.find_adsorption_sites()
    
    target_sites = []
    if site_type == "all":
        target_sites = sites_dict.get('ontop', []) + sites_dict.get('bridge', []) + sites_dict.get('hollow', [])
    else:
        target_sites = sites_dict.get(site_type, [])
        
    if not target_sites:
        return f"Error: No {site_type} sites found. Available types: {list(sites_dict.keys())}"
    
    # Parse repeat pattern
    try:
        repeat_list = [int(x.strip()) for x in repeat.split(',')]
        if len(repeat_list) != 3:
            return "Error: repeat must be 'a,b,c' format (e.g., '1,1,1' or '2,2,1')."
    except ValueError:
        return "Error: repeat values must be integers (e.g., '1,1,1')."
    
    # Generate structures for all sites
    generated_files = []
    base_name, ext = os.path.splitext(output_path)
    if not ext:
        ext = "" # Handle case where no extension
    
    for i, site in enumerate(target_sites):
        ads_struct = asf.add_adsorbate(mol, site, repeat=repeat_list)
        
        # Construct filename with index
        # If output_path is "POSCAR", result is "POSCAR_0", "POSCAR_1"
        # If output_path is "structure.vasp", result is "structure_0.vasp"
        current_output_path = f"{base_name}_{i}{ext}"
        
        ads_struct.to(filename=current_output_path)
        generated_files.append(current_output_path)
    
    return (f"Adsorbates added. Generated {len(generated_files)} structures:\n"
            f"  Adsorbate: {adsorbate}\n"
            f"  Site type: {site_type}\n"
            f"  Height: {height} Å\n"
            f"  Repeat: {repeat}\n"
            f"  Files:\n" + "\n".join([f"    - {f}" for f in generated_files]))


def substitute_element(
    input_path: Annotated[
        str,
        "Path to the input structure file."
    ],
    output_path: Annotated[
        str,
        "Path where the modified structure will be saved."
    ],
    original_element: Annotated[
        str,
        "Element symbol to be replaced (e.g., 'Fe', 'O')."
    ],
    new_element: Annotated[
        str,
        "New element symbol to substitute (e.g., 'Co', 'S')."
    ],
    fraction: Annotated[
        float,
        "Fraction of atoms to substitute (0.0-1.0). Use 1.0 for all atoms."
    ] = 1.0,
    random_seed: Annotated[
        Optional[int],
        "Random seed for reproducible partial substitution. None for random."
    ] = None
) -> str:
    """
    Substitute one element with another in the structure.
    
    This tool replaces atoms of one element with another element.
    Can perform full or partial (random) substitution.
    
    Args:
        input_path: Path to input structure file.
        output_path: Path for output modified structure.
        original_element: Element to be replaced.
        new_element: New element to substitute.
        fraction: Fraction of atoms to replace (0.0-1.0).
        random_seed: Seed for reproducible random selection.
    
    Returns:
        str: Success message with substitution details.
    
    Example:
        substitute_element("POSCAR", "POSCAR_doped", "Fe", "Co", 0.5)
    """
    from pymatgen.core import Structure, Element
    import random
    
    structure = Structure.from_file(input_path)
    
    # Validate elements
    try:
        orig_elem = Element(original_element)
        new_elem = Element(new_element)
    except ValueError as e:
        return f"Error: Invalid element symbol. {e}"
    
    # Find indices of original element
    indices = [i for i, site in enumerate(structure) if site.specie.symbol == original_element]
    
    if not indices:
        return f"Error: Element '{original_element}' not found in structure."
    
    # Determine how many to substitute
    n_substitute = int(len(indices) * fraction)
    if n_substitute == 0 and fraction > 0:
        n_substitute = 1  # At least one if fraction > 0
    
    # Random selection for partial substitution
    if fraction < 1.0:
        if random_seed is not None:
            random.seed(random_seed)
        indices_to_replace = random.sample(indices, n_substitute)
    else:
        indices_to_replace = indices
    
    # Perform substitution
    modified = structure.copy()
    for idx in indices_to_replace:
        modified.replace(idx, new_elem)
    
    modified.to(filename=output_path)
    
    return (f"Element substitution completed at {output_path}\n"
            f"  {original_element} → {new_element}\n"
            f"  Substituted: {len(indices_to_replace)}/{len(indices)} atoms\n"
            f"  Fraction: {fraction*100:.1f}%\n"
            f"  Total atoms: {len(modified)}")


def create_structure_workbench() -> StaticWorkbench:
    """
    Create a workbench with all structure manipulation tools.
    
    Returns:
        StaticWorkbench: Workbench containing supercell, slab, adsorbate,
                        and substitution tools.
    """
    # Create FunctionTools for each function
    supercell_tool = FunctionTool(make_supercell, description=make_supercell.__doc__)
    slab_tool = FunctionTool(make_slab, description=make_slab.__doc__)
    adsorbate_tool = FunctionTool(add_adsorbate, description=add_adsorbate.__doc__)
    substitute_tool = FunctionTool(substitute_element, description=substitute_element.__doc__)
    return StaticWorkbench([
        supercell_tool,
        slab_tool,
        adsorbate_tool,
        substitute_tool
    ])


__all__ = [
    "make_supercell",
    "make_slab", 
    "add_adsorbate",
    "substitute_element",
    "create_structure_workbench",
]


if __name__ == '__main__':
    # Test the tools
    print("Structure Tools Module")
    print("=" * 50)
    print("\nAvailable tools:")
    print("  1. make_supercell - Create supercell structures")
    print("  2. make_slab - Generate surface slabs")
    print("  3. add_adsorbate - Add molecules to surfaces")
    print("  4. substitute_element - Replace elements")
    print("\nUsage:")
    print("  from tools.structure_tools import create_structure_workbench")
    print("  workbench = create_structure_workbench()")

