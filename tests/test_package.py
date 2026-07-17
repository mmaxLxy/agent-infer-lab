def test_package_exposes_version() -> None:
    import agent_infer_lab

    assert agent_infer_lab.__version__ == "0.1.0"
