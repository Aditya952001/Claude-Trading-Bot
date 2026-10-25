import pathlib

OLD = "from numba import njit\n"
NEW = (
    "try:\n"
    "    from numba import njit\n"
    "except ImportError:  # numba unavailable on Python 3.14+; use no-op\n"
    "    def njit(*args, **kwargs):\n"
    "        def _w(fn): return fn\n"
    "        return args[0] if (len(args)==1 and callable(args[0])) else _w\n"
)

import sys
root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".venv/lib/python3.14/site-packages/pandas_ta")
count = 0
for f in root.rglob("*.py"):
    txt = f.read_text(encoding="utf-8")
    if OLD in txt and "except ImportError" not in txt:
        f.write_text(txt.replace(OLD, NEW, 1), encoding="utf-8")
        print("Patched:", f)
        count += 1
print(f"Done. {count} files patched.")
