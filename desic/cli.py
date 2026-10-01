"""Command line entry point: ``desic serve`` / ``desic demo`` / ``desic eval``."""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    # Legacy Windows code pages (e.g. cp1254) cannot encode "→" and friends; degrade instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(prog="desic", description="Self-learning, explainable decision engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the API and realtime dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data", default=os.environ.get("DESIC_DATA", "data"), help="data directory (SQLite + models)")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")

    demo = sub.add_parser("demo", help="create demo questions (support tickets + loan applications) to explore")
    demo.add_argument("--data", default=os.environ.get("DESIC_DATA", "data"))
    demo.add_argument("--rows", type=int, default=600, help="training tickets (loan applications: 3x)")

    ev = sub.add_parser("eval", help="run the evaluation scenarios (Banking77: noise, drift, forgetting, teacher …)")
    from .eval.run import add_arguments

    add_arguments(ev)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        os.environ["DESIC_DATA"] = args.data
        print(f"Desic dashboard → http://{args.host}:{args.port}   (API docs: /docs)")
        uvicorn.run("desic.api:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    elif args.cmd == "demo":
        from .demo import build_demo

        print(build_demo(args.data, args.rows))
    elif args.cmd == "eval":
        from .eval.run import run

        run(args)


if __name__ == "__main__":
    main()
