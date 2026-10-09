#!/usr/bin/env python3
"""Compatibility entrypoint for the real Wan student training worker."""

from tools.wan_student_train_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
