import pytest

def test_tui_import_and_init():
    """Basic smoke test for RaAidTUI instantiation."""
    try:
        from ra_aid.tui import RaAidTUI
    except ImportError as e:
        pytest.fail(f"Could not import RaAidTUI: {e}")

    # Instantiate (do not run event loop)
    try:
        app = RaAidTUI()
    except Exception as e:
        pytest.fail(f"Could not instantiate RaAidTUI: {e}")