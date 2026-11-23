"""Cloth (2D flex) point grasps and the private `cloth` metric (Slurm test node)."""
from pathlib import Path
import tomllib

import numpy as np
import pytest

from sim2sim import cloth
from sim2sim.evaluation import selected_metrics
from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'width': 320, 'height': 240, 'graspable_bodies': [], 'timestep': .002}


def test_camera_pixel_grasp_lifts_a_cloth_corner_and_reports_its_weight():
    sim = Simulation(ROOT / 'environments/cloth_quadrants/scene.xml', {**CONFIG, 'wrench': True})
    try:
        assert sim._body_names == [] and len(set(sim._constraints.values())) == 1  # one shared flex connect
        bodies = cloth._Cloth(sim).bodies
        corner = sim._data.xpos[bodies][np.argmax(sim._data.xpos[bodies, 0] - sim._data.xpos[bodies, 1])] + [-.008, .008, 0]
        cam = sim._camera_id
        local = sim._data.cam_xmat[cam].reshape(3, 3).T @ (corner - sim._data.cam_xpos[cam])
        scale = np.tan(np.deg2rad(sim._model.cam_fovy[cam]) / 2)
        pixel = [(local[0] / -local[2] / scale * sim.height / sim.width + 1) * sim.width / 2 - .5, (1 - local[1] / -local[2] / scale) * sim.height / 2 - .5]
        assert sim.action({'op': 'grasp', 'pixel': pixel}, render=False) is None and sim._grasp is not None
        sim.action({'op': 'move', 'displacement': [0, 0, .9], 'ticks': 900}, render=False)
        sim.action({'op': 'wait', 'ticks': 750}, render=False)
        weight = sim._model.body_mass[bodies].sum() * 9.81
        assert sim._data.xpos[bodies, 2].min() > .45  # the whole cloth (0.57 m diagonal) hangs clear of the table
        assert sim._wrench()['force'][2] == pytest.approx(-weight, rel=.05)
    finally:
        sim.close()


def test_cloth_metric_is_zero_against_itself_and_sees_stiffness(tmp_path):
    spec = tomllib.loads((ROOT / 'environments/cloth_quadrants/manifest.toml').read_text())['cloth']
    spec = {**spec, 'drops': spec['drops'][:1] + spec['drops'][-1:], 'drop_ticks': 300}
    source, base = ROOT / 'environments/cloth_quadrants/scene.xml', ROOT / 'environments/cloth_base/scene.xml'
    same = cloth.evaluate(source, source, CONFIG, CONFIG, spec, tmp_path / 'same')
    other = cloth.evaluate(source, base, CONFIG, CONFIG, spec, tmp_path / 'other')
    assert same['status'] == other['status'] == 'ok' and same['stiffness_map']['probed_points'] == 16
    assert same['rest_chamfer_m'] == 0 and same['stiffness_map']['lift_force_mae_n'] == 0 and same['drop_mean_chamfer_m'] == 0
    assert other['rest_chamfer_m'] < .005 and other['stiffness_map']['lift_force_relative_error'] > .08 and other['drop_final_chamfer_m'] > .005
    assert (tmp_path / 'other/drop_001/side_by_side.mp4').stat().st_size > 0 and (tmp_path / 'other/stiffness_map.png').is_file()
    assert 'cloth' in selected_metrics({'metrics': ['cloth', 'image_mse']})
