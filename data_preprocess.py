from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch
atom_dict=defaultdict(lambda: len(atom_dict))
bond_dict=defaultdict(lambda: len(bond_dict))
fingerprint_dict=defaultdict(lambda: len(fingerprint_dict))
edge_dict=defaultdict(lambda:len(edge_dict))
radius=1

#function for getting atoms from the smiles string
def create_atoms(mol,atom_dict):
  atoms=[a.GetSymbol() for a  in mol.GetAtoms()]
  for a in mol.GetAromaticAtoms():
    i=a.GetIdx()
    atoms[i]=(atoms[i],'aromatic')
  atoms=[atom_dict[a] for a in atoms]
  return np.array(atoms)

#function for creating a dictionary of bonds with each bond type assigned a unique hash value
def create_ijbonddict(mol,bond_dict):
  i_jbond_dict=defaultdict(lambda: [])
  for b in mol.GetBonds():
    i,j=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    bond=bond_dict[str(b.GetBondType())]
    i_jbond_dict[i].append((j,bond))
    i_jbond_dict[j].append((i,bond))
  return i_jbond_dict

#function for extracting fingerprints from the atoms and the bond_dict

def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.。
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)




