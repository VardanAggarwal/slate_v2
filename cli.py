"""Commands: encode | consolidate | replay | digest | rebuild | stats. See PLAN.md §7."""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slate-engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encode", help="encode a note and print the receipt")
    p_enc.add_argument("--text", help="note text (or pipe via stdin)")
    p_enc.add_argument("--file", help="read note text from a file")
    p_enc.add_argument("--title", default=None)

    p_rep = sub.add_parser("replay", help="replay the old slate.db corpus")
    p_rep.add_argument("--old-db", default=None)
    p_rep.add_argument("--limit", type=int, default=None)

    sub.add_parser("stats", help="store counts + last consolidation run")
    sub.add_parser("consolidate", help="nightly sleep phase (Phase 2)")
    sub.add_parser("digest", help="morning digest (Phase 6)")
    sub.add_parser("rebuild", help="rebuild semantic store from event log (Phase 2)")

    args = ap.parse_args(argv)

    if args.cmd == "encode":
        from core import store
        from core.encode import encode
        if args.file:
            text = open(args.file).read()
        elif args.text:
            text = args.text
        else:
            text = sys.stdin.read()
        receipt = encode(store.connect(), text, title=args.title)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))

    elif args.cmd == "replay":
        from migrate import replay
        replay(old_db=args.old_db, limit=args.limit)

    elif args.cmd == "stats":
        from core import store
        print(json.dumps(store.stats(store.connect()), indent=2))

    else:
        print(f"`{args.cmd}` lands in a later phase (see PLAN.md §7)")
        sys.exit(1)


if __name__ == "__main__":
    main()
