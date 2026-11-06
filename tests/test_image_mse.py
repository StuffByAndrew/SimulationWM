"""Private RGB MSE math and trusted reset rendering regressions."""
import io
from pathlib import Path
from unittest.mock import patch
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from sim2sim.evaluation import _rgb_mse, evaluate_submission, selected_metrics


def test_normalized_full_image_rgb_mse():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0] = [255, 0, 0]
    assert _rgb_mse(a, b) == pytest.approx(1 / 12)
    assert _rgb_mse(b, a) == pytest.approx(1 / 12)
    assert _rgb_mse(a, np.full_like(a, 255)) == 1
    assert selected_metrics({'metrics': ['image_mse']}) == {'image_mse'}


@pytest.mark.parametrize('selector', ['image_mse', 'appearance'])
def test_private_mse_uses_trusted_camera_and_full_image(tmp_path, selector):
    source = Path(__file__).resolve().parents[1] / 'environments/blocks'
    text = (source / 'scene.xml').read_text()
    def evaluate(candidate):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('scene.xml', candidate)
        return evaluate_submission(data.getvalue(), {
            'source_dir': str(source), 'output_dir': str(tmp_path),
            'width': 64, 'height': 48, 'metrics': [selector],
            'source_config': {'rgb': False, 'depth': False, 'calibration': False}})
    with patch('sim2sim.evaluation._probe', side_effect=AssertionError('disabled dynamics')):
        tree = ET.fromstring(text)
        for camera in tree.iter('camera'):
            camera.set('pos', '7 8 9')
            camera.set('fovy', '15')
        same = evaluate(ET.tostring(tree, encoding='unicode'))
        changed = evaluate(text.replace('size=".06 .06 .06"', 'size=".11 .11 .11"'))
    assert same['status'] == changed['status'] == 'ok', (same, changed)
    assert same['scores']['reset']['rgb_mse'] == pytest.approx(0, abs=1e-12)
    assert 0 < changed['scores']['reset']['rgb_mse'] <= 1
    assert set(same['scores']) == {'reset'}
    if selector == 'image_mse':
        assert set(same['scores']['reset']) == {'rgb_mse'}
