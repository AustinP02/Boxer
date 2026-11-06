"""Compatibility entry point for the read-only color/depth detection preview."""

from boxer.cli import main

if __name__ == "__main__":
    main(["preview", "--frames", "100"])
