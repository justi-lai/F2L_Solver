import torch
import os
import tqdm
import sys
import random
import argparse
from multiprocessing import Pool, cpu_count, Manager

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.cube_generator import CubeGenerator

def worker_func(args):
    """
    Worker function to generate a single sample.
    Args:
        args: Tuple (dagger_prob, seed, optimize)
    """
    dagger_prob, seed, optimize = args
    random.seed(seed)
    gen = CubeGenerator()
    
    try:
        if dagger_prob > 0 and random.random() < dagger_prob:
             states, actions = gen.generate_dagger_solve(scramble_length=25, epsilon=0.1, optimize=optimize)
        else:
             states, actions = gen.generate_solve(scramble_length=25, optimize=optimize)
             
        entry = {
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.long)
        }
        return entry
    except Exception as e:
        print(f"Error in worker: {e}", file=sys.stderr)
        return None

def generate_dataset(num_samples=100, output_path="data/dataset", dagger_prob=0.5, workers=None, chunk_size=10000, optimize=False):
    """
    Generates a dataset of solves and saves it as sharded .pt files.
    """
    if workers is None:
        workers = cpu_count()
        
    # Ensure output is a directory
    base_name, ext = os.path.splitext(output_path)
    if ext == ".pt":
        # User passed file path, convert to dir
        output_dir = base_name
    else:
        output_dir = output_path
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} samples into {output_dir} (Chunk Size: {chunk_size}, Optimize: {optimize})...")
    
    # Pass optimize flag to workers
    worker_args = [(dagger_prob, random.randint(0, 10**9), optimize) for _ in range(num_samples)]
    
    buffer = []
    chunk_idx = 0
    total_saved = 0
    
    with Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(worker_func, worker_args, chunksize=10)
        
        with tqdm.tqdm(total=num_samples) as pbar:
            for result in iterator:
                if result is not None:
                    buffer.append(result)
                pbar.update(1)
                
                # Check buffer size
                if len(buffer) >= chunk_size:
                    save_path = os.path.join(output_dir, f"part_{chunk_idx}.pt")
                    torch.save(buffer, save_path)
                    total_saved += len(buffer)
                    buffer = [] # Clear memory
                    chunk_idx += 1
                    
    # Save remainder
    if buffer:
        save_path = os.path.join(output_dir, f"part_{chunk_idx}.pt")
        torch.save(buffer, save_path)
        total_saved += len(buffer)
        
    print(f"Done. Saved {total_saved} samples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for Rubik's Cube solving.")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate.")
    parser.add_argument("--dagger_prob", type=float, default=0.5,
                        help="Probability of using DAgger for sample generation.")
    parser.add_argument("--output_path", type=str, default="data/dataset",
                        help="path to save the generated dataset directory.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes.")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Number of samples per shard file.")
    parser.add_argument("--optimize", action="store_true",
                        help="If set, generates OPTIMIZED solves (fewer rotations). Default is False (Unoptimized).")
    
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.num_samples, 
        dagger_prob=args.dagger_prob, 
        output_path=args.output_path,
        workers=args.workers,
        chunk_size=args.chunk_size,
        optimize=args.optimize
    )
