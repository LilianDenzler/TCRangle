
import numpy as np
from pymol import cmd, cgo
cmd.load("/workspaces/Graphormer/TRangle/data/consensus_output/chain_B/average_structure.pdb","average_structure")
cmd.bg_color("white")
cmd.hide("everything","all")
cmd.show("cartoon","average_structure")
cmd.color("gray70","average_structure")
cmd.set("cartoon_transparency", 0.3, "average_structure")

# Show centroid
cmd.pseudoatom("centroid_average_structure", pos=[np.float32(99.76387), np.float32(-25.82735), np.float32(-73.88838)], color="red")
cmd.show("spheres","centroid_average_structure")
cmd.set("sphere_scale", 0.6, "centroid_average_structure")

# Create CGO arrows for PCA axes
cmd.load_cgo([
        cgo.CYLINDER,99.764,-25.827,-73.888,
                     82.127,-23.970,-64.642,
                     0.3,
                     0.2,0.5,1.0,
                     0.2,0.5,1.0,
        cgo.CONE,82.127,-23.970,-64.642,
                 99.764,-25.827,-73.888,
                 0.44999999999999996,0.0,
                 0.2,0.5,1.0,
                 0.2,0.5,1.0,1.0
    ], "PC1_average_structure")
cmd.load_cgo([
        cgo.CYLINDER,99.764,-25.827,-73.888,
                     97.973,-45.738,-73.304,
                     0.3,
                     0.1,0.8,0.1,
                     0.1,0.8,0.1,
        cgo.CONE,97.973,-45.738,-73.304,
                 99.764,-25.827,-73.888,
                 0.44999999999999996,0.0,
                 0.1,0.8,0.1,
                 0.1,0.8,0.1,1.0
    ], "PC2_average_structure")

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png("/workspaces/Graphormer/TRangle/data/consensus_output/chain_B/average_structure_pca_vis.png", dpi=300, ray=1)
cmd.save("/workspaces/Graphormer/TRangle/data/consensus_output/chain_B/average_structure_pca_vis.pse")
cmd.quit()
