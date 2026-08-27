from __future__ import annotations

from racing.robots import list_racing_candidates


def main() -> None:
    rows = list_racing_candidates()
    print(f"{'MODEL':18s} {'nu':>4s} {'nv':>4s} {'nav':>4s} {'stadium':>8s}  root")
    for r in rows:
        if 'error' in r:
            print(f"{r['name']:18s} ERROR: {r['error']}")
            continue
        print(
            f"{r['name']:18s} {r['nu']:4d} {r['nv']:4d} {r['navigation']:>4s} "
            f"{str(r['stadium']):>8s}  {r['root_body']}"
        )


if __name__ == '__main__':
    main()
