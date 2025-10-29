"""Debug script for CONCAT+FRAME test case."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.canvas import (
    infer_canvas,
    _enumerate_concat_candidates,
    _verify_canvas_map,
    _apply_concat,
    _apply_frame,
)


def test_concat_frame_debug():
    """Debug CONCAT + FRAME composition."""
    X = [[1, 2]]
    Y = [
        [9, 9, 9, 9],
        [9, 1, 2, 9],
        [9, 1, 2, 9],
        [9, 9, 9, 9]
    ]

    print("Input X:", X)
    print("Output Y:")
    for row in Y:
        print("  ", row)

    print("\n--- Step 1: Enumerate CONCAT candidates ---")
    concat_candidates = _enumerate_concat_candidates([(X, Y)])
    print(f"Found {len(concat_candidates)} CONCAT candidates:")
    for i, cand in enumerate(concat_candidates[:10]):  # Limit output
        print(f"  {i+1}. axis={cand.axis}, k={cand.k}, gap={cand.gap}, gap_color={cand.gap_color}")

    print("\n--- Step 2: Try CONCAT alone ---")
    for cand in concat_candidates[:5]:
        temp = _apply_concat(X, cand.axis, cand.k, cand.gap, cand.gap_color or 0)
        print(f"  axis={cand.axis}, k={cand.k}, gap={cand.gap} → size {len(temp)}x{len(temp[0])}")
        verified = _verify_canvas_map(cand, [(X, Y)])
        print(f"    Verified: {verified}")

    print("\n--- Step 3: Try CONCAT + FRAME ---")
    # Manually try the expected case
    temp = _apply_concat(X, "rows", 2, 0, 0)
    print(f"After CONCAT (rows, k=2, gap=0):")
    for row in temp:
        print("  ", row)

    final = _apply_frame(temp, 9, 1)
    print(f"\nAfter FRAME (color=9, thickness=1):")
    for row in final:
        print("  ", row)

    print(f"\nExpected Y:")
    for row in Y:
        print("  ", row)

    print(f"\nMatches: {final == Y}")

    print("\n--- Step 4: Run full inference ---")
    result = infer_canvas([(X, Y)])
    print(f"Result: {result}")


if __name__ == "__main__":
    test_concat_frame_debug()
