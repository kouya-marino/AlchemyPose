from alchemypose.inference.session import COREML, CPU, chosen_providers


def test_coreml_is_requested_first_when_preferred_and_available() -> None:
    assert chosen_providers(True, [COREML, CPU]) == [COREML, CPU]


def test_cpu_only_when_coreml_is_not_wanted() -> None:
    assert chosen_providers(False, [COREML, CPU]) == [CPU]


def test_cpu_only_when_coreml_is_unavailable() -> None:
    assert chosen_providers(True, [CPU]) == [CPU]


def test_cpu_is_always_last_so_there_is_always_a_fallback() -> None:
    assert chosen_providers(True, [COREML, CPU])[-1] == CPU
    assert chosen_providers(False, [COREML, CPU])[-1] == CPU
