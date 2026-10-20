from hypothesis import HealthCheck, settings

# HealthCheck.too_slow fails a test when Hypothesis draws fewer than ~10 valid
# inputs in its first second of wall-clock time. Under pytest-xdist on a cold
# checkout, the first property test of a worker shares the CPU with the other
# workers compiling the numba cache, and draws that take 1 ms warm take over
# 100 ms, so whichever property test runs first failed at random. Generation
# speed is not what these tests check; the deadline on each example still is.
settings.register_profile("default", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("default")
