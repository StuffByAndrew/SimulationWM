"""The private `gear_train` metric (Slurm test node): coupling-aware gear removal and the shallow reference's scores."""
import io
from pathlib import Path
import tomllib
import zipfile

import pytest

from sim2sim.evaluation import evaluate_submission
from sim2sim.gear_train import compile_plain, remove_body
from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / 'environments/gear_train'
_SIM = tomllib.loads((ROOT / 'experiments/configs/wave28-20261124/w28-gear-recon.toml').read_text())['simulation']
GRASP = {'grasp_slip_m': _SIM['grasp_slip_m']}  # the level's public grip slip

PAIR = '''<mujoco><worldbody><camera name="fixed" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
  <body name="a"><joint name="ja" axis="0 1 0"/><geom size=".02"/></body>
  <body name="b" pos=".1 0 0"><joint name="jb" axis="0 1 0"/><geom size=".02"/><body><joint name="jc"/><geom size=".01"/></body></body>
</worldbody>
<equality><joint joint1="jb" joint2="ja" polycoef="0 -1 0 0 0"/><joint joint1="jc" joint2="ja"/><tendon tendon1="t"/></equality>
<tendon><fixed name="t"><joint joint="jb" coef="1"/></fixed></tendon>
<actuator><motor joint="jb"/><motor joint="ja"/></actuator>
<sensor><jointpos joint="jc"/></sensor>
<keyframe><key qpos="0 0 0"/></keyframe></mujoco>'''


def _zip(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('scene.xml', text)
    return buffer.getvalue()


def test_removing_a_body_drops_its_subtree_and_everything_coupled_to_it(tmp_path):
    (tmp_path / 'scene.xml').write_text(PAIR)
    model, _ = compile_plain(tmp_path / 'scene.xml')
    sim = Simulation(tmp_path / 'scene.xml', {'width': 64, 'height': 48, 'scene_edit': remove_body(model, 2)})
    try:
        m = sim._model
        assert [m.body(i).name for i in range(m.nbody) if not m.body(i).name.startswith('_grasp_')] == ['world', 'a']
        assert m.ntendon == m.nsensor == m.nkey == 0 and m.nu == 1  # ja's motor stays
        assert [m.eq(i).name for i in range(m.neq)] == ['_grasp_connect_0']
    finally:
        sim.close()


def test_grasp_softness_knob_is_validated_and_sets_the_grasp_constraint(tmp_path):
    (tmp_path / 'scene.xml').write_text(PAIR)
    for bad in ({'grasp_solimp': [.99, .999, .001]}, {'grasp_solref': [.002, float('nan')]}, {'grasp_solref': '0.002 1'}):
        with pytest.raises(ValueError, match='finite numbers'):
            Simulation(tmp_path / 'scene.xml', {'width': 64, 'height': 48, **bad})
    stiff = {'grasp_solref': [.002, 1], 'grasp_solimp': [.99, .999, .001, .5, 2]}
    sim = Simulation(tmp_path / 'scene.xml', {'width': 64, 'height': 48, **stiff})
    try:
        eq = sim._constraints[1][0]
        assert list(sim._model.eq_solref[eq]) == pytest.approx(stiff['grasp_solref'])
        assert list(sim._model.eq_solimp[eq]) == pytest.approx(stiff['grasp_solimp'])
    finally:
        sim.close()
    default = Simulation(tmp_path / 'scene.xml', {'width': 64, 'height': 48})
    try:
        assert list(default._model.eq_solimp[default._constraints[1][0]]) == pytest.approx([.9, .95, .001, .5, 2])
    finally:
        default.close()


def test_grip_slips_only_when_stretched_past_grasp_slip_m(tmp_path):
    (tmp_path / 'scene.xml').write_text(PAIR)
    for bad in (0, -.01, 2, True, '0.004', float('inf')):
        with pytest.raises(ValueError, match='grasp_slip_m'):
            Simulation(tmp_path / 'scene.xml', {'width': 64, 'height': 48, 'grasp_slip_m': bad})
    for slip in (None, .005):
        sim = Simulation(tmp_path / 'scene.xml', {'width': 64, 'height': 48, **({'grasp_slip_m': slip} if slip else {})})
        try:
            sim._grasp_controller.attach(1, sim._data.xpos[1].copy())  # a point on a's axle: no move can carry it
            sim.action({'op': 'move', 'displacement': [.003, 0, 0], 'ticks': 20}, render=False)
            assert sim._grasp is not None  # stretched 3 mm: holds
            sim.action({'op': 'move', 'displacement': [.01, 0, 0], 'ticks': 20}, render=False)
            assert (sim._grasp is None) == bool(slip) and sim.observe()['grasped'] is not bool(slip)
            if slip:
                assert not sim._data.eq_active[sim._constraints[1][0]]
                with pytest.raises(ValueError, match='active grasp'):
                    sim.action({'op': 'move', 'displacement': [.001, 0, 0], 'ticks': 1}, render=False)
        finally:
            sim.close()


def test_shallow_reference_is_shallow_passes_the_gate_and_cannot_answer_removals(tmp_path):
    config = {'source_dir': str(LEVEL), 'output_dir': str(tmp_path), 'metrics': ['gear_train'], 'camera': 'fixed', 'width': 320, 'height': 240, **GRASP}
    scores = evaluate_submission(_zip((LEVEL / 'shallow.xml').read_text()), config)['scores']['gear_train']
    assert scores['structure']['depth_class'] == 'shallow' and scores['structure']['matched'] == 0
    assert scores['accuracy']['candidate_timestep_s'] == scores['sim_cost']['shallow_reference']['timestep_s'] > .0005  # own step
    assert scores['cheap']['status'] == 'ok' and scores['cheap']['cost_relative_to_shallow'] > 0
    assert scores['removal']['cannot_answer'] == len(scores['removal']['gears']) and scores['removal']['mean_probe_error_rad'] is None
    assert (tmp_path / 'artifacts').is_dir()
    unnamed = evaluate_submission(_zip((LEVEL / 'shallow.xml').read_text().replace('name="hand"', 'name="dial"')), {**config, 'output_dir': str(tmp_path / 'b')})
    assert unnamed['status'] == 'failed' and 'hand' in unnamed['error']
