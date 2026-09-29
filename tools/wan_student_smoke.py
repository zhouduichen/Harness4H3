#!/usr/bin/env python3
"""Generate a short clip from the real pretrained Wan2.1 Student."""

import json
import sys

from harness4h3.student.wan import WanStudentError, main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WanStudentError as exc:
        print(json.dumps({"ok": False, "error_code": exc.code, "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
