"""Launch the local humanoid operator console."""


def main() -> None:
    # Spawned node processes also import this entry point.
    from humanoid.api.server import main as serve_console  # noqa: PLC0415

    serve_console()


if __name__ == "__main__":
    main()
