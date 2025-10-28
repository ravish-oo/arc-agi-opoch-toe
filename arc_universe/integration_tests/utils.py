"""
Utility functions for ARC-AGI integration testing.

Provides:
- Data loading from JSON challenge files
- Grid type conversion
- Random task sampling
- Receipt generation
- Logging setup
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Type aliases matching arc_core
Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]


def get_data_dir() -> Path:
    """Get the absolute path to the data directory."""
    # arc_universe/integration_tests/utils.py -> arc_universe -> parent -> data
    return Path(__file__).parent.parent.parent / "data"


def load_arc_tasks(
    dataset: str = "training", limit: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load ARC tasks from JSON challenge files.

    Args:
        dataset: One of "training", "evaluation", "test"
        limit: Optional limit on number of tasks to load

    Returns:
        Dict mapping task_id -> task_dict with 'train' and 'test' keys

    Raises:
        FileNotFoundError: If challenge file doesn't exist
        ValueError: If dataset name is invalid
    """
    valid_datasets = {"training", "evaluation", "test"}
    if dataset not in valid_datasets:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Must be one of {valid_datasets}"
        )

    data_dir = get_data_dir()
    challenge_file = data_dir / f"arc-agi_{dataset}_challenges.json"

    if not challenge_file.exists():
        raise FileNotFoundError(f"Challenge file not found: {challenge_file}")

    with open(challenge_file, "r") as f:
        all_tasks = json.load(f)

    # Apply limit if specified
    if limit is not None:
        task_ids = list(all_tasks.keys())[:limit]
        all_tasks = {tid: all_tasks[tid] for tid in task_ids}

    return all_tasks


def random_sample_tasks(
    dataset: str = "training", n: int = 50, seed: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load a random sample of N tasks from the specified dataset.

    Args:
        dataset: One of "training", "evaluation", "test"
        n: Number of tasks to sample
        seed: Optional random seed for reproducibility

    Returns:
        Dict mapping task_id -> task_dict
    """
    # Load all tasks
    all_tasks = load_arc_tasks(dataset=dataset, limit=None)
    task_ids = list(all_tasks.keys())

    # Sample N tasks
    if seed is not None:
        random.seed(seed)

    sampled_ids = random.sample(task_ids, min(n, len(task_ids)))

    return {tid: all_tasks[tid] for tid in sampled_ids}


def parse_task(task_dict: Dict[str, Any]) -> Tuple[List[TrainPair], List[Grid]]:
    """
    Parse a task dictionary into train pairs and test inputs.

    Args:
        task_dict: Task dict with 'train' and 'test' keys

    Returns:
        Tuple of (train_pairs, test_inputs) where:
        - train_pairs: List of (input_grid, output_grid) tuples
        - test_inputs: List of input grids
    """
    train_pairs = [
        (example["input"], example["output"]) for example in task_dict["train"]
    ]

    test_inputs = [example["input"] for example in task_dict["test"]]

    return train_pairs, test_inputs


def grid_to_type(grid_list: List[List[int]]) -> Grid:
    """
    Convert JSON grid (list of lists) to Grid type.

    This is a no-op in the current implementation since Grid = List[List[int]],
    but exists for future type compatibility.

    Args:
        grid_list: Grid as nested lists from JSON

    Returns:
        Grid in the expected type
    """
    return grid_list


def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """
    Setup logger for integration tests.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def build_receipt(
    task_id: str,
    gate: str,
    present_data: Optional[Dict[str, Any]] = None,
    wl_data: Optional[Dict[str, Any]] = None,
    shape_data: Optional[Dict[str, Any]] = None,
    status: str = "PASS",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a receipt dictionary for a task.

    Args:
        task_id: Task identifier
        gate: Gate name ("A", "B", etc.)
        present_data: Present validation data
        wl_data: WL validation data
        shape_data: Shape validation data (Gate B+)
        status: "PASS" or "FAIL"
        error: Error message if status is FAIL

    Returns:
        Receipt dictionary
    """
    receipt = {
        "task_id": task_id,
        "gate": gate,
        "timestamp": datetime.now().isoformat(),
        "status": status,
    }

    if present_data is not None:
        receipt["present"] = present_data

    if wl_data is not None:
        receipt["wl"] = wl_data

    if shape_data is not None:
        receipt["shape"] = shape_data

    if error is not None:
        receipt["error"] = error

    return receipt


def save_receipt(receipt: Dict[str, Any], output_dir: Path) -> None:
    """
    Save receipt to JSON file.

    Args:
        receipt: Receipt dictionary
        output_dir: Directory to save receipt (e.g., receipts/gate_a/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = receipt["task_id"]
    receipt_file = output_dir / f"{task_id}.json"

    with open(receipt_file, "w") as f:
        json.dump(receipt, f, indent=2)


def compute_summary_stats(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from a list of receipts.

    Args:
        receipts: List of receipt dictionaries

    Returns:
        Summary statistics dictionary
    """
    total = len(receipts)
    passed = sum(1 for r in receipts if r["status"] == "PASS")

    stats = {
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0.0,
    }

    # WL statistics
    if any("wl" in r for r in receipts):
        wl_receipts = [r for r in receipts if "wl" in r]
        iters = [r["wl"]["iters"] for r in wl_receipts]
        unseen_violations = sum(
            1 for r in wl_receipts if r["wl"]["unseen_roles"] > 0
        )

        stats["wl"] = {
            "avg_iterations": sum(iters) / len(iters) if iters else 0.0,
            "max_iterations": max(iters) if iters else 0,
            "unseen_roles_violations": unseen_violations,
            "determinism_pass_rate": sum(
                1 for r in wl_receipts if r["wl"].get("deterministic", False)
            )
            / len(wl_receipts)
            if wl_receipts
            else 0.0,
        }

    # Shape statistics
    if any("shape" in r for r in receipts):
        shape_receipts = [r for r in receipts if "shape" in r]
        row_classes = [r["shape"]["num_row_classes"] for r in shape_receipts]
        col_classes = [r["shape"]["num_col_classes"] for r in shape_receipts]

        stats["shape"] = {
            "avg_row_classes": sum(row_classes) / len(row_classes)
            if row_classes
            else 0.0,
            "avg_col_classes": sum(col_classes) / len(col_classes)
            if col_classes
            else 0.0,
        }

    return stats
