#!/usr/bin/env python3
"""LMSV 统一 CLI 入口，支持所有 Task 的执行。"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_conf():
    import genconf
    if hasattr(genconf, 'main'):
        genconf.main()
    else:
        os.system(f"cd {REPO_ROOT} && python genconf.py")


def run_do():
    import do
    do.main()


def run_analyze(args):
    import analyze
    if hasattr(analyze, 'main'):
        analyze.main(args[2:] if len(args) > 2 else [])
    else:
        os.system(f"cd {REPO_ROOT} && python analyze.py")


def run_repro():
    import repro
    if hasattr(repro, 'main'):
        repro.main()
    else:
        os.system(f"cd {REPO_ROOT} && python repro.py")


def run_slave():
    import slave
    if hasattr(slave, 'main'):
        slave.main()
    else:
        os.system(f"cd {REPO_ROOT} && python slave.py")


def run_webui():
    os.system(f"cd {REPO_ROOT} && python webui.py")


def show_help():
    print("Usage: lmsv [command]")
    print("")
    print("Commands:")
    print("  webui   Run web UI")
    print("  conf    Generate config (interactive)")
    print("  do      Execute task from config.json")
    print("  slave   Run slave service for multi-node mode")
    print("  analyze Regenerate analysis")
    print("  repro   Reproduce a single run")
    print("  help    Show this help message")
    print("")
    print("No command: run conf + do")


def main():
    if len(sys.argv) < 2:
        run_conf()
        run_do()
        return 0

    cmd = sys.argv[1]
    if cmd == "conf":
        run_conf()
    elif cmd == "do":
        run_do()
    elif cmd == "analyze":
        run_analyze(sys.argv)
    elif cmd == "repro":
        run_repro()
    elif cmd == "slave":
        run_slave()
    elif cmd == "webui":
        run_webui()
    elif cmd in ("help", "-h", "--help"):
        show_help()
    else:
        print(f"Unknown command: {cmd}")
        show_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
