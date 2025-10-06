## THIS REPOSITORY IS DEPRECATED ##
# Please see https://github.com/LilianDenzler/trangle-pip

# Instructions for using the new package:

## Installation
This package can be installed via
```bash
pip install tcrgeometry
```

### Extra dependencies

For TCR numbering, `ANARCII` must be installed via pip: (using the newest version)

```bash
pip install anarcii
```
Sometimes the legacy version ANARCI works better, so it must also be installed via conda:
```bash
conda install bioconda::anarci
```
For visualization, `PyMOL` must be installed via bioconda:

```bash
conda install conda-forge::pymol-open-source
```


## Measure angles of existing TCR structures
To measure angles in existing TCR structures, you can use the `new_calc.py` script provided in the TRangle package. This script allows you to calculate angles and distances in a TCR structure file.


```bash
tcr-calc --input_pdb file.pdb --out_path ./out
```
Add the --vis flag to generate a visualization of the modified structure. It will save a visualisation of the measured angles and distance saved as an image and a .pse file which can be opened in PyMOL.

or with the API:
```python
import tcrgeometry
calc_results = tcrgeometry.calc_tcr_geometry(pdb_file, out_path="./out")
```
Set the vis=True flag to generate a visualization of the modified structure.



This will output a CSV file with the measured angles and distances.

## Measure angles of existing TCR trajectories
To measure angles in existing TCR trajectories, you can use the `new_calc_MD.py` script provided in the TRangle package. This script allows you to calculate angles and distances in a TCR trajectory file.

```bash
tcr-calc-md --input_pdb file.pdb --input_md trajectory.traj --out_path ./out
```
or with the API:
```python
import tcrgeometry
md_results = tcrgeometry.calc_tcr_geometry_MD(pdb_file, md_path, out_path="./out")
```

## Change geometry of a TCR structure

To change the geometry of a TCR structure, you can use the `change_geometry.py` script provided in the TRangle package. This script allows you to modify angles and distances in a TCR structure based on a configuration file.

```bash
tcr-change --input file.pdb --out_path ./out --BA 113 --BC1 98.7 --BC2 9.3 --AC1 71.5 --AC2 154 --dc 24
```
Add the --vis flag to generate a visualization of the modified structure.


or with the API:
```python
import tcrgeometry
changed_pdb=tcrgeometry.change_tcr_geometry(pdb_file, out_path="./out", BA=113, BC1=98.7, BC2=9.3, AC1=71.5, AC2=154, dc=24)
```
Set the vis=True flag to generate a visualization of the modified structure.

This script will read the configuration file, apply the specified changes to the angles and distances, and output a new PDB file with the modified geometry.

## Extract loop anchor residue coordinates
To extract the coordinates of loop anchor residues from a TCR structure, you can use the `extract_loop_anchor.py` script provided in the TRangle package. This script allows you to specify the loop anchor residues and extract their coordinates from a TCR structure file.

```bash
tcr-extract-anchor path/to/your/input.pdb
```
or with the API:
```python
import tcrgeometry
anchor_coords = tcrgeometry.get_anchor_coords(pdb_file)
```
This will output a CSV file containing the coordinates of the specified loop anchor residues, which can be used for input to the CDR loop diffusion model.



Dataset:
From STCRDB get non-redundant abTCR set of IMGT-numbered structures (resolution cutoff 3.0, sequence identity cutoff 70%)
