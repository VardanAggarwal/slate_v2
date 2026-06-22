"""Commands: encode | consolidate | replay | digest | rebuild | stats | user. See PLAN.md §7, AUTH.md §6.

Multi-user: data commands take --user (a username or raw user_id; default
DEFAULT_USER_ID for local dev). consolidate and digest default to ALL users so
the nightly cron lines (DEPLOY.md §7) keep working unchanged.
"""
import argparse
import getpass
import json
import sys


def _resolve_user(conn, value: str) -> str:
    """Accept a username (resolved via the users table) or a raw user_id."""
    from core import store
    row = store.get_user_by_username(conn, value)
    return row["id"] if row else value


def main(argv=None):
    from core import config

    ap = argparse.ArgumentParser(prog="slate-engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_user_arg(p, default=config.DEFAULT_USER_ID):
        p.add_argument("--user", default=default,
                       help="username or user_id (default: %(default)s)")

    p_enc = sub.add_parser("encode", help="encode a note and print the receipt")
    p_enc.add_argument("--text", help="note text (or pipe via stdin)")
    p_enc.add_argument("--file", help="read note text from a file")
    p_enc.add_argument("--title", default=None)
    add_user_arg(p_enc)

    p_rep = sub.add_parser("replay", help="replay the old slate.db corpus")
    p_rep.add_argument("--old-db", default=None)
    p_rep.add_argument("--limit", type=int, default=None)
    add_user_arg(p_rep)

    p_sta = sub.add_parser("stats", help="store counts + last consolidation run")
    add_user_arg(p_sta, default=None)

    p_con = sub.add_parser("consolidate", help="nightly sleep phase")
    p_con.add_argument("--max-episodes", type=int, default=50)
    p_con.add_argument("--all", action="store_true",
                       help="loop until no unconsolidated episodes remain")
    add_user_arg(p_con, default=None)  # default: every user with pending episodes

    p_ref = sub.add_parser("refine", help="Write-side refine pass (W2–W8): "
                                          "fragment + route unfragmented episodes")
    p_ref.add_argument("--max-episodes", type=int, default=200)
    add_user_arg(p_ref, default=None)  # default: every user with pending episodes

    p_dig = sub.add_parser("digest", help="morning digest from recent events")
    p_dig.add_argument("--since-hours", type=int, default=36)
    p_dig.add_argument("--polish", action="store_true", help="LLM prose pass")
    add_user_arg(p_dig, default=None)  # default: every user

    p_rec = sub.add_parser("reconstruct", help="regenerate a note from its blueprint")
    p_rec.add_argument("episode_id")
    add_user_arg(p_rec)

    sub.add_parser("rebuild", help="rebuild semantic store from event log (all users)")

    p_usr = sub.add_parser("user", help="manage users (AUTH.md §6)")
    usr_sub = p_usr.add_subparsers(dest="user_cmd", required=True)
    u_add = usr_sub.add_parser("add", help="create a user (password prompted)")
    u_add.add_argument("username")
    u_add.add_argument("--admin", action="store_true")
    usr_sub.add_parser("list", help="list users")
    u_pwd = usr_sub.add_parser("passwd", help="reset a user's password")
    u_pwd.add_argument("username")
    u_rm = usr_sub.add_parser("rm", help="remove a user's login (corpus stays)")
    u_rm.add_argument("username")

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
        from core import write
        conn = store.connect()
        user_id = _resolve_user(conn, args.user)
        receipt = encode(conn, user_id, text, title=args.title, source="cli")
        # CLI is not latency-sensitive: run W2–W8 inline so a single encode yields
        # the full pipeline (the MCP path defers this to a background thread).
        refined = write.refine_episode(conn, user_id, receipt["episode_id"])
        print(json.dumps({**receipt, "refine": refined}, indent=2, ensure_ascii=False))

    elif args.cmd == "replay":
        from core import store
        from migrate import replay
        conn = store.connect()
        user_id = _resolve_user(conn, args.user)
        conn.close()
        replay(user_id, old_db=args.old_db, limit=args.limit)

    elif args.cmd == "stats":
        from core import store
        conn = store.connect()
        if args.user:
            print(json.dumps(store.stats(conn, _resolve_user(conn, args.user)), indent=2))
        else:
            out = {uid: store.stats(conn, uid) for uid in store.all_user_ids(conn)}
            print(json.dumps(out, indent=2))

    elif args.cmd == "refine":
        from core import store, write
        conn = store.connect()
        target = _resolve_user(conn, args.user) if args.user else None
        print(json.dumps(write.refine_pending(conn, target,
                                              max_episodes=args.max_episodes),
                         indent=2, ensure_ascii=False))

    elif args.cmd == "consolidate":
        from core import store, write
        from core.consolidate import consolidate, consolidate_all_users
        conn = store.connect()
        # Write-side catch-up before the sleep phase (cron's `consolidate --all`
        # line thus also drains the refine queue — no separate cron entry needed).
        write.refine_pending(conn, _resolve_user(conn, args.user) if args.user else None)
        if args.user:
            user_id = _resolve_user(conn, args.user)
            while True:
                report = consolidate(conn, user_id, max_episodes=args.max_episodes)
                print(json.dumps(report, indent=2))
                if not args.all or report["status"] == "noop":
                    break
        else:
            while True:
                reports = consolidate_all_users(conn, max_episodes=args.max_episodes)
                print(json.dumps(reports, indent=2))
                progress = any(r.get("episodes", 0) > 0 for r in reports)
                if not args.all or not reports or not progress:
                    break

    elif args.cmd == "digest":
        from core import store
        from core.digest import digest
        conn = store.connect()
        if args.user:
            print(digest(conn, _resolve_user(conn, args.user),
                         since_hours=args.since_hours, polish=args.polish))
        else:
            for uid in store.all_user_ids(conn):
                user = store.get_user(conn, uid)
                name = user["username"] if user else uid
                print(f"## digest — {name}\n")
                print(digest(conn, uid, since_hours=args.since_hours,
                             polish=args.polish))
                print()

    elif args.cmd == "reconstruct":
        from core import store
        from core.reconstruct import reconstruct
        conn = store.connect()
        print(json.dumps(reconstruct(conn, _resolve_user(conn, args.user),
                                     args.episode_id),
                         indent=2, ensure_ascii=False))

    elif args.cmd == "rebuild":
        from core import store
        from core.consolidate import rebuild
        print(json.dumps(rebuild(store.connect()), indent=2))

    elif args.cmd == "user":
        from core import store
        from core.auth import hash_password
        conn = store.connect()
        if args.user_cmd == "add":
            if store.get_user_by_username(conn, args.username):
                sys.exit(f"user exists: {args.username}")
            pw = getpass.getpass(f"Password for {args.username}: ")
            if not pw or pw != getpass.getpass("Repeat: "):
                sys.exit("empty password or mismatch")
            with conn:
                user_id = store.create_user(conn, args.username, hash_password(pw),
                                            is_admin=args.admin)
            print(f"created {user_id} ({args.username}"
                  f"{', admin' if args.admin else ''})")
        elif args.user_cmd == "list":
            for u in store.list_users(conn):
                print(f"{u['id']}  {u['username']:24s}  "
                      f"{'admin' if u['is_admin'] else 'user':5s}  "
                      f"episodes={store.count_episodes(conn, u['id'])}  "
                      f"created={u['created_at'][:10]}")
        elif args.user_cmd == "passwd":
            if not store.get_user_by_username(conn, args.username):
                sys.exit(f"no such user: {args.username}")
            pw = getpass.getpass(f"New password for {args.username}: ")
            if not pw or pw != getpass.getpass("Repeat: "):
                sys.exit("empty password or mismatch")
            with conn:
                store.set_user_password(conn, args.username, hash_password(pw))
            print("password updated")
        elif args.user_cmd == "rm":
            with conn:
                if not store.delete_user(conn, args.username):
                    sys.exit(f"no such user: {args.username}")
            print(f"removed login {args.username} (corpus data retained)")

    else:
        print(f"`{args.cmd}` lands in a later phase (see PLAN.md §7)")
        sys.exit(1)


if __name__ == "__main__":
    main()
