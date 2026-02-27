def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate indicator hash snapshots (tests/fixtures/snapshots.json)",
    )
