"""Command line entry point: ``desic serve`` / ``desic demo``."""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="desic", description="Self-learning, explainable decision engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the API and realtime dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data", default=os.environ.get("DESIC_DATA", "data"), help="data directory (SQLite + models)")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")

    demo = sub.add_parser("demo", help="create a demo model and dataset so the dashboard has something to show")
    demo.add_argument("--data", default=os.environ.get("DESIC_DATA", "data"))
    demo.add_argument("--rows", type=int, default=2000)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        os.environ["DESIC_DATA"] = args.data
        print(f"Desic dashboard → http://{args.host}:{args.port}   (API docs: /docs)")
        uvicorn.run("desic.api:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    elif args.cmd == "demo":
        from .demo import build_demo

        print(build_demo(args.data, args.rows))


if __name__ == "__main__":
    main()
