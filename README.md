## Installation
This package can be installed via
```bash
pip install tcrgeometry
```
Alternatively, clone the repository and install locally.

### Set up environment
The file `environment.yml` specifies the dependencies required to run the TRangle package. You can create a conda environment using this file:

```bash
conda env create -f environment.yml
```
In the main project directory run
```bash
pip install -e .
```
## Usage Instructions for TRangle
### Measure angles of existing TCR structures
To measure angles in existing TCR structures, you can use the `new_calc.py` script provided in the TRangle package. This script allows you to calculate angles and distances in a TCR structure file.


```bash
python trangle/calc_geometry.py --input_pdb path/to/your/input.pdb
```
This will output a CSV file with the measured angles and distances.
It will also output a PDB of the extracted variable domain, was well as a visualiseation of the measured angles and distance saved as an image and a .pse file which can be opened in PyMOL.


### Measure angles of existing TCR trajectories
To measure angles in existing TCR trajectories, you can use the `new_calc_MD.py` script provided in the TRangle package. This script allows you to calculate angles and distances in a TCR trajectory file.

```bash
python trangle/calc_geometry_MD.py --input_pdb path/to/your/input.pdb --input_md path/to/your/input.traj
```

### Change geometry of a TCR structure

To change the geometry of a TCR structure, you can use the `change_geometry.py` script provided in the TRangle package. This script allows you to modify angles and distances in a TCR structure based on a configuration file.

```bash
python trangle/change_geometry.py --input_pdb path/to/your/input.pdb --BA 113.22 --BC1 98.75 --BC2 9.35 --AC1 71.58 --AC2 154.62 --dc 23.98
```
This script will read the configuration file, apply the specified changes to the angles and distances, and output a new PDB file with the modified geometry. It will also generate a visualization of the modified structure for inspection.

### Extract loop anchor residue coordinates
To extract the coordinates of loop anchor residues from a TCR structure, you can use the `extract_loop_anchor.py` script provided in the TRangle package. This script allows you to specify the loop anchor residues and extract their coordinates from a TCR structure file.

```bash
python trangle/get_anchor_coords.py path/to/your/input.pdb
```
This will output a CSV file containing the coordinates of the specified loop anchor residues, which can be used for input to the CDR loop diffusion model.



Dataset:
From STCRDB get non-redundant abTCR set of IMGT-numbered structures (resolution cutoff 3.0, sequence identity cutoff 70%)