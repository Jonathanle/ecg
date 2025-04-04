# ecg_adapter.py
# Module for adapting ECG data files to a specific format required by the analysis pipeline

import os
import re
import shutil
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ecg_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_ecg_file(filepath: Path) -> bool:
    """
    Validates if the given file has the expected ECG data format (250, 12)
    
    Args:
        filepath: Path to the ECG xlsx file
        
    Returns:
        bool: True if the file has the expected format, False otherwise
    """
    try:
        df = pd.read_excel(filepath, header=None, index_col=None)
        if df.shape == (250, 12):
            return True
        else:
            logger.warning(f"File {filepath} has unexpected shape: {df.shape} instead of (250, 12)")
            return False
    except Exception as e:
        logger.error(f"Error validating file {filepath}: {str(e)}")
        return False

def extract_numeric_id(filename: str) -> Optional[str]:
    """
    Extracts the numeric identifier from a filename
    
    Args:
        filename: Filename to extract numeric ID from
        
    Returns:
        str: Extracted numeric ID or None if not found
    """
    # Extract digits from filename (assumes main identifier is numeric)
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    else:
        logger.warning(f"Could not extract numeric ID from filename: {filename}")
        return None

def create_adapted_file(source_file: Path, output_dir: Path, 
                      prefix: str = "R", suffix: str = "_anon_interp", 
                      file_type: str = "pre") -> Optional[Path]:
    """
    Creates a new file with adapted naming convention
    
    Args:
        source_file: Path to the source ECG file
        output_dir: Directory to save the adapted file
        prefix: Prefix to add to the numeric ID (default: "R")
        suffix: Suffix to add after the file type (default: "_anon_interp")
        file_type: Type of file ("pre" or "post")
        
    Returns:
        Path: Path to the created file or None if creation failed
    """
    try:
        # Extract numeric ID from the filename
        numeric_id = extract_numeric_id(source_file.name)
        if not numeric_id:
            return None
            
        # Construct new filename
        new_filename = f"{prefix}{numeric_id}_{file_type}{suffix}.xlsx"
        target_path = output_dir / new_filename
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy the file with the new name
        shutil.copy2(source_file, target_path)
        logger.info(f"Created {file_type} file: {target_path}")
        
        return target_path
    except Exception as e:
        logger.error(f"Error creating adapted file from {source_file}: {str(e)}")
        return None

def process_ecg_file(source_file: Path, output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Processes a single ECG file, creating both pre and post versions
    
    Args:
        source_file: Path to the source ECG file
        output_dir: Directory to save the adapted files
        
    Returns:
        Tuple of Paths to the created pre and post files, or None if creation failed
    """
    # Validate the file
    if not validate_ecg_file(source_file):
        logger.warning(f"Skipping invalid file: {source_file}")
        return None, None
    
    # Create pre and post files
    pre_file = create_adapted_file(source_file, output_dir, file_type="pre")
    post_file = create_adapted_file(source_file, output_dir, file_type="post")
    
    return pre_file, post_file

def convert_ecg_file_format(source_dir: str, output_dir: str, 
                           subdirs: List[str] = ["GEN", "MYO", "SARC"],
                           use_parallel: bool = True,
                           max_workers: int = 4) -> Dict[str, int]:
    """
    Converts ECG files from source directory to the required format in output directory
    
    Args:
        source_dir: Directory containing source ECG files
        output_dir: Directory to save adapted ECG files
        subdirs: List of subdirectories to process
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of worker processes
        
    Returns:
        Dict with statistics about the conversion
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    stats = {
        "total_files": 0,
        "converted_files": 0,
        "failed_files": 0,
        "processed_subdirs": []
    }
    
    logger.info(f"Starting conversion from {source_dir} to {output_dir}")
    
    # Find all xlsx files in specified subdirectories
    ecg_files = []
    for subdir in subdirs:
        subdir_path = source_path / subdir
        if not subdir_path.exists():
            logger.warning(f"Subdirectory {subdir_path} does not exist")
            continue
            
        stats["processed_subdirs"].append(subdir)
        for file_path in subdir_path.glob("*.xlsx"):
            ecg_files.append(file_path)
            stats["total_files"] += 1
    
    logger.info(f"Found {len(ecg_files)} ECG files in {len(stats['processed_subdirs'])} subdirectories")
    
    # Process files in parallel or sequentially
    if use_parallel and len(ecg_files) > 10:
        logger.info(f"Using parallel processing with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_ecg_file, file_path, output_path) for file_path in ecg_files]
            for future in futures:
                try:
                    pre_file, post_file = future.result()
                    if pre_file and post_file:
                        stats["converted_files"] += 1
                    else:
                        stats["failed_files"] += 1
                except Exception as e:
                    logger.error(f"Error in parallel processing: {str(e)}")
                    stats["failed_files"] += 1
    else:
        logger.info("Using sequential processing")
        for file_path in ecg_files:
            pre_file, post_file = process_ecg_file(file_path, output_path)
            if pre_file and post_file:
                stats["converted_files"] += 1
            else:
                stats["failed_files"] += 1
    
    # Log summary
    success_rate = (stats["converted_files"] / stats["total_files"]) * 100 if stats["total_files"] > 0 else 0
    logger.info(f"Conversion complete. Success rate: {success_rate:.2f}%")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Converted files: {stats['converted_files']}")
    logger.info(f"Failed files: {stats['failed_files']}")
    
    return stats

def verify_adapted_files(output_dir: str) -> Dict[str, int]:
    """
    Verifies that files in the output directory match the expected format
    
    Args:
        output_dir: Directory containing adapted files
        
    Returns:
        Dict with verification statistics
    """
    output_path = Path(output_dir)
    
    stats = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "pre_files": 0,
        "post_files": 0
    }
    
    logger.info(f"Verifying adapted files in {output_dir}")
    
    # Check all xlsx files in the output directory
    for file_path in output_path.glob("*.xlsx"):
        stats["total_files"] += 1
        
        # Check naming convention
        if "_pre_" in file_path.name:
            stats["pre_files"] += 1
        elif "_post_" in file_path.name:
            stats["post_files"] += 1
        
        # Validate file content
        if validate_ecg_file(file_path):
            stats["valid_files"] += 1
        else:
            stats["invalid_files"] += 1
    
    # Log verification summary
    logger.info(f"Verification complete.")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Valid files: {stats['valid_files']}")
    logger.info(f"Invalid files: {stats['invalid_files']}")
    logger.info(f"Pre files: {stats['pre_files']}")
    logger.info(f"Post files: {stats['post_files']}")
    
    return stats

def main():
    """
    Main function for running the ECG file adapter directly
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ECG files to required format')
    parser.add_argument('--source', required=True, help='Source directory containing ECG files')
    parser.add_argument('--output', required=True, help='Output directory for adapted files')
    parser.add_argument('--subdirs', nargs='+', default=["GEN", "MYO", "SARC"], 
                        help='Subdirectories to process')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--verify', action='store_true', help='Verify adapted files after conversion')
    
    args = parser.parse_args()
    
    # Run conversion
    stats = convert_ecg_file_format(
        args.source, 
        args.output, 
        subdirs=args.subdirs,
        use_parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Verify if requested
    if args.verify:
        verify_stats = verify_adapted_files(args.output)
        
        # Check if verification matches conversion
        if verify_stats["valid_files"] != stats["converted_files"] * 2:  # *2 because each converted file creates 2 files
            logger.warning("Verification counts don't match conversion counts. Some files may be missing or invalid.")

if __name__ == "__main__":
    main()
