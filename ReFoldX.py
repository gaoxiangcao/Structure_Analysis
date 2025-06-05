import argparse
import pymol2
from pyrosetta import *
from Bio import pairwise2
from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue
from pyrosetta.rosetta.core.chemical import VariantType
from pyrosetta.rosetta.protocols.relax import FastRelax


class StructureStitcher:
    def __init__(self, af3_pdb, fragment_pdbs, output_pdb):
        self.af3_pdb = af3_pdb
        self.fragment_pdbs = fragment_pdbs
        self.output_pdb = output_pdb

        # Initialize PyRosetta with fixed seed for reproducibility
        init("-mute all -constant_seed -jran 12345")

        # Load full model
        self.full_pose = pose_from_pdb(af3_pdb)

    def align_fragments(self):
        aligned_paths = []
        for frag_path in self.fragment_pdbs:
            aligned_path = frag_path.replace(".pdb", "_aligned.pdb")
            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                cmd.load(frag_path, "frag")
                cmd.load(self.af3_pdb, "target")
                cmd.super("frag", "target")
                cmd.delete("target")

                # Force creation of new aligned object
                # reload the cords of the atoms...
                # Todo: the reason of this bug ...
                cmd.create("frag_aligned", "frag")
                cmd.delete("frag")
                cmd.save(aligned_path, "frag_aligned")

            # Re-save with PyRosetta to ensure clean structure (no multiple MODELs)
            pose_cleaned = pose_from_pdb(aligned_path)
            pose_cleaned.dump_pdb(aligned_path)

            aligned_paths.append(aligned_path)
        return aligned_paths

    def insert_by_sequence(self, target_pose, fragment_pose):
        target_seq = target_pose.sequence()
        frag_seq = fragment_pose.sequence()
        alignments = pairwise2.align.localms(target_seq, frag_seq, 2, -1, -10, -1)
        if not alignments:
            raise ValueError("No alignment found between target and fragment")

        aln = alignments[0]
        start = aln.start + 1
        end = aln.end
        max_res = min(fragment_pose.total_residue(), end - start + 1)

        for i in range(1, fragment_pose.total_residue() + 1):
            remove_variant_type_from_pose_residue(fragment_pose, VariantType.LOWER_TERMINUS_VARIANT, i)
            remove_variant_type_from_pose_residue(fragment_pose, VariantType.UPPER_TERMINUS_VARIANT, i)

        for i in range(max_res):
            target_pose.replace_residue(start + i, fragment_pose.residue(i + 1), orient_backbone=True)

    def run(self):
        stitched = Pose()
        stitched.assign(self.full_pose)

        aligned_fragments = self.align_fragments()
        for frag_aligned in aligned_fragments:
            frag_pose = pose_from_pdb(frag_aligned)
            self.insert_by_sequence(stitched, frag_pose)

        stitched.dump_pdb("stitched_tmp.pdb")
        stitched = pose_from_pdb("stitched_tmp.pdb")  # Rebuild chain connections

        relax = FastRelax()
        relax.set_scorefxn(get_fa_scorefxn())
        relax.apply(stitched)

        stitched.dump_pdb(self.output_pdb)
        print(f"Stitched and relaxed model saved to: {self.output_pdb}")


def main():
    parser = argparse.ArgumentParser(description="Stitch AF3 structure with experimental fragments.")
    parser.add_argument("--af3", required=True, help="Full-length AF3 model PDB")
    parser.add_argument("--fragments", required=True, nargs="+", help="List of fragment PDB files to stitch in")
    parser.add_argument("--output", required=True, help="Output stitched model PDB filename")
    args = parser.parse_args()

    stitcher = StructureStitcher(args.af3, args.fragments, args.output)
    stitcher.run()


if __name__ == "__main__":
    main()
    print("Run successfully!")