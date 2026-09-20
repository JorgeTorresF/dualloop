from dualloop.calibration import ConfidenceCalibrator


def test_cold_start_returns_close_to_raw():
    calib = ConfidenceCalibrator()
    assert abs(calib.calibrate(0.9) - 0.9) < 0.15


def test_calibration_corrects_overconfidence_with_evidence():
    calib = ConfidenceCalibrator(cold_start_n=5.0)
    # el motor reporta 0.9 de confianza pero se equivoca sistematicamente
    for _ in range(50):
        calib.update(0.9, correct=False)
    calibrated = calib.calibrate(0.9)
    assert calibrated < 0.3, f"esperaba correccion fuerte hacia abajo, obtuve {calibrated}"


def test_calibration_confirms_good_confidence_with_evidence():
    calib = ConfidenceCalibrator(cold_start_n=5.0)
    for _ in range(50):
        calib.update(0.9, correct=True)
    calibrated = calib.calibrate(0.9)
    assert calibrated > 0.85


def test_state_dict_roundtrip():
    calib = ConfidenceCalibrator()
    for _ in range(10):
        calib.update(0.4, correct=True)
    state = calib.state_dict()

    restored = ConfidenceCalibrator()
    restored.load_state_dict(state)
    assert restored.calibrate(0.4) == calib.calibrate(0.4)
