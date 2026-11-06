"""One entry point for Boxer tools. Example: python boxer.py sort watch --preview"""
import argparse
import runpy
import sys
from pathlib import Path

TOOLS = {"pickup": "pickup_demo.py", "alignment": "check_alignment.py", "sort": "color_sorter.py", "feed": "live_feed.py",
         "scan": "sample_scan.py", "arc": "inspection_arc.py",
         "analyze": "analyze_media.py", "templates": "build_shape_templates.py", "green": "green_to_container.py", "pill": "pill_to_container.py"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=TOOLS)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.args
    source = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source))
    script = source / TOOLS[args.tool]
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
