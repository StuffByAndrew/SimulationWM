"""The private `gear_train` metric (Slurm test node): coupling-aware gear removal and the shallow reference's scores."""
import io
from pathlib import Path
import zipfile

from sim2sim.evaluation import evaluate_submission
from sim2sim.gear_train import compile_plain, remove_body
from sim2sim.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / 'environments/gear_train'

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


def test_shallow_reference_is_shallow_passes_the_gate_and_cannot_answer_removals(tmp_path):
    config = {'source_dir': str(LEVEL), 'output_dir': str(tmp_path), 'metrics': ['gear_train'], 'camera': 'fixed', 'width': 320, 'height': 240}
    scores = evaluate_submission(_zip((LEVEL / 'shallow.xml').read_text()), config)['scores']['gear_train']
    assert scores['structure']['depth_class'] == 'shallow' and scores['structure']['matched'] == 0
    assert scores['cheap']['status'] == 'ok' and scores['cheap']['cost_relative_to_shallow'] > 0
    assert scores['removal']['cannot_answer'] == len(scores['removal']['gears']) and scores['removal']['mean_probe_error_rad'] is None
    assert (tmp_path / 'artifacts').is_dir()
    unnamed = evaluate_submission(_zip((LEVEL / 'shallow.xml').read_text().replace('name="hand"', 'name="dial"')), {**config, 'output_dir': str(tmp_path / 'b')})
    assert unnamed['status'] == 'failed' and 'hand' in unnamed['error']
