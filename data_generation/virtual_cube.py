import os
import sys
import torch
import numpy as np

# Add PyCube-Solver to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PyCube-Solver"))
from cube import Cube

class VirtualCube:
    """
    A wrapper for the PyCube object that produces a relative, rotation-invariant
    tensor representation of the cube state.
    """
    def __init__(self, cube_instance):
        """
        Args:
            cube_instance (Cube): An instance of the PyCube-Solver Cube class.
        """
        self.cube_obj = cube_instance
        
        # Define the face indices based on PyCube-Solver/cube.py
        # 0:F, 1:R, 2:B, 3:L, 4:D, 5:U
        self.F, self.R, self.B, self.L, self.D, self.U = 0, 1, 2, 3, 4, 5
        
        # Define the Slot Definitions
        # Each slot is a tuple of (Face, Row, Col) coordinates for each face involved.
        # Order matters for orientation definition.
        # Edges: Primary Face, Secondary Face
        # Corners: Primary, Secondary, Tertiary
        
        # 12 Edge Slots (Indices 0-11)
        self.edge_slots = [
            # UF (Slot 0) - U(2,1), F(0,1)
            ((self.U, 2, 1), (self.F, 0, 1)),
            # UR (Slot 1) - U(1,2), R(0,1)
            ((self.U, 1, 2), (self.R, 0, 1)),
            # UB (Slot 2) - U(0,1), B(0,1)
            ((self.U, 0, 1), (self.B, 0, 1)),
            # UL (Slot 3) - U(1,0), L(0,1)
            ((self.U, 1, 0), (self.L, 0, 1)),
            # DF (Slot 4) - D(0,1), F(2,1)
            ((self.D, 0, 1), (self.F, 2, 1)),
            # DR (Slot 5) - D(1,2), R(2,1)
            ((self.D, 1, 2), (self.R, 2, 1)),
            # DB (Slot 6) - D(2,1), B(2,1)
            ((self.D, 2, 1), (self.B, 2, 1)),
            # DL (Slot 7) - D(1,0), L(2,1)
            ((self.D, 1, 0), (self.L, 2, 1)),
            # FR (Slot 8) - F(1,2), R(1,0)
            ((self.F, 1, 2), (self.R, 1, 0)),
            # FL (Slot 9) - F(1,0), L(1,2)
            ((self.F, 1, 0), (self.L, 1, 2)),
            # BR (Slot 10) - B(1,0), R(1,2)
            ((self.B, 1, 0), (self.R, 1, 2)),
            # BL (Slot 11) - B(1,2), L(1,0)
            ((self.B, 1, 2), (self.L, 1, 0)),
        ]
        
        # 8 Corner Slots (Indices 0-7 for Corners, or 12-19 in global)
        self.corner_slots = [
            # UFR (Slot 0) - U(2,2), F(0,2), R(0,0)
            ((self.U, 2, 2), (self.F, 0, 2), (self.R, 0, 0)),
            # URB (Slot 1) - U(0,2), R(0,2), B(0,0)
            ((self.U, 0, 2), (self.R, 0, 2), (self.B, 0, 0)),
            # UBL (Slot 2) - U(0,0), B(0,2), L(0,0)
            ((self.U, 0, 0), (self.B, 0, 2), (self.L, 0, 0)),
            # ULF (Slot 3) - U(2,0), L(0,2), F(0,0)
            ((self.U, 2, 0), (self.L, 0, 2), (self.F, 0, 0)),
            # DRF (Slot 4) - D(0,2), R(2,0), F(2,2)
            ((self.D, 0, 2), (self.R, 2, 0), (self.F, 2, 2)),
            # DFL (Slot 5) - D(0,0), F(2,0), L(2,2)
            ((self.D, 0, 0), (self.F, 2, 0), (self.L, 2, 2)),
            # DLB (Slot 6) - D(2,0), L(2,0), B(2,2)
            ((self.D, 2, 0), (self.L, 2, 0), (self.B, 2, 2)),
            # DBR (Slot 7) - D(2,2), B(2,0), R(2,2)
            ((self.D, 2, 2), (self.B, 2, 0), (self.R, 2, 2)),
        ]
        
    def get_centers(self):
        """
        Returns the current colors of the centers of the 6 faces.
        Returns:
            dict: {face_index: color_string}
        """
        centers = {}
        for f in range(6):
            centers[f] = self.cube_obj.cube[f][1][1]
        return centers

    def get_one_hot_tensor(self):
        """
        Generates the 20x24 one-hot tensor representing the relative state.
        
        Rows 0-11: Edges (UF, UR, UB, UL, DF, DR, DB, DL, FR, FL, BR, BL)
        Rows 12-19: Corners (UFR, URB, UBL, ULF, DRF, DFL, DLB, DBR)
        
        Columns:
        For Edges: Slot_Index (0-11) * 2 + Orientation (0-1)
        For Corners: Slot_Index (0-7) * 3 + Orientation (0-2)
        """
        centers = self.get_centers()
        
        # Initialize tensor
        # 20 rows (pieces), 24 cols (positions)
        tensor = torch.zeros(20, 24)
        
        # --- Handle Edges ---
        for piece_idx, target_slot in enumerate(self.edge_slots):
            # 1. Define Target Colors for this piece (in Primary, Secondary order)
            target_c1 = centers[target_slot[0][0]]
            target_c2 = centers[target_slot[1][0]]
            
            # 2. Scan all physical edge slots to find this piece
            found = False
            for loc_idx, current_slot in enumerate(self.edge_slots):
                # Read colors at this physical slot
                c1 = self.cube_obj.cube[current_slot[0][0]][current_slot[0][1]][current_slot[0][2]]
                c2 = self.cube_obj.cube[current_slot[1][0]][current_slot[1][1]][current_slot[1][2]]
                
                # Check match
                # Case A: Correct Orientation
                if c1 == target_c1 and c2 == target_c2:
                    orientation = 0
                    col_idx = loc_idx * 2 + orientation
                    tensor[piece_idx, col_idx] = 1.0
                    found = True
                    break
                # Case B: Flipped Orientation
                elif c1 == target_c2 and c2 == target_c1:
                    orientation = 1
                    col_idx = loc_idx * 2 + orientation
                    tensor[piece_idx, col_idx] = 1.0
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Edge Piece {piece_idx} ({target_c1}{target_c2}) not found! Invalid cube state?")

        # --- Handle Corners ---
        for piece_idx_offset, target_slot in enumerate(self.corner_slots):
            piece_idx = 12 + piece_idx_offset
            
            # 1. Define Target Colors
            target_c1 = centers[target_slot[0][0]]
            target_c2 = centers[target_slot[1][0]]
            target_c3 = centers[target_slot[2][0]]
            target_set = {target_c1, target_c2, target_c3}
            
            # 2. Scan all physical corner slots
            found = False
            for loc_idx, current_slot in enumerate(self.corner_slots):
                # Read colors
                c1 = self.cube_obj.cube[current_slot[0][0]][current_slot[0][1]][current_slot[0][2]]
                c2 = self.cube_obj.cube[current_slot[1][0]][current_slot[1][1]][current_slot[1][2]]
                c3 = self.cube_obj.cube[current_slot[2][0]][current_slot[2][1]][current_slot[2][2]]
                
                current_set = {c1, c2, c3}
                
                if current_set == target_set:
                    # Found the piece! Now determine orientation.
                    if c1 == target_c1:
                        orientation = 0
                    elif c1 == target_c2: 
                        orientation = 1 
                    elif c1 == target_c3:
                        orientation = 2
                    else:
                        raise ValueError("Logic Error in Corner Orientation")
                        
                    col_idx = loc_idx * 3 + orientation
                    tensor[piece_idx, col_idx] = 1.0
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Corner Piece {piece_idx} not found!")
                
        return tensor

    def __str__(self):
        """
        Returns the string representation of the physical cube state.
        This is useful for manually validating moves.
        """
        return str(self.cube_obj)

if __name__ == "__main__":
    # Test Verification
    print("Initializing Cube...")
    c = Cube()
    vc = VirtualCube(c)
    
    print("Generating Solved State Tensor...")
    t = vc.get_one_hot_tensor()
    print("Shape:", t.shape)
    
    # Verify Solved State (Should be diagonal-ish)
    actual_sum = t.sum().item()
    print(f"Sum: {actual_sum} (Expected 20.0)")
    assert actual_sum == 20.0
    
    if t[0, 0] == 1: print("Piece UF Correct")
    else: print("Piece UF Incorrect")
    
    solved_tensor = t.clone()

    # --- Test 1: Rotation Invariance ---
    print("\n--- Test 1: Rotation Invariance (y move) ---")
    c.doMoves("y")
    t_rot = vc.get_one_hot_tensor()
    
    if torch.equal(t_rot, solved_tensor):
        print("PASS: Tensor unchanged after rotation (Relative view works).")
    else:
        print("FAIL: Tensor changed after rotation!")
        
    # Reset
    c = Cube()
    vc = VirtualCube(c)

    # --- Test 2: Multi-move Scramble ---
    print("\n--- Test 2: Multi-move Scramble (R U R' U') ---")
    scramble = "RURPUP" 
    c.doMoves(scramble)
    t_scramble = vc.get_one_hot_tensor()
    
    print(f"Scramble Tensor Sum: {t_scramble.sum().item()}")
    assert t_scramble.sum().item() == 20.0
    
    if not torch.equal(t_scramble, solved_tensor):
        print("PASS: Scramble produced different state than solved.")
    else:
        print("FAIL: Scramble did not change state!")

    # Visualize Scrambled State
    print("\nScrambled Cube State:")
    print(vc)
    
    print("\nVerification Complete.")
