"""Metric-family selectors skip work, not just returned fields."""
import io
from pathlib import Path
import zipfile
from unittest.mock import patch
import pytest
from sim2sim.evaluation import selected_metrics, evaluate_submission


def test_selector_validation_is_pure():
    assert selected_metrics({'metrics':['appearance']})=={'appearance'}
    assert 'dynamics' in selected_metrics({})
    for value in ('appearance',['unknown'],[True]):
        with pytest.raises(ValueError):selected_metrics({'metrics':value})


def test_appearance_only_never_executes_other_families(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=root/'environments/blocks';bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w') as archive:archive.writestr('scene.xml',(source/'scene.xml').read_bytes())
    with patch('sim2sim.evaluation._probe',side_effect=AssertionError('disabled dynamics executed')), \
         patch('sim2sim.evaluation._parameters',side_effect=AssertionError('disabled parameters executed')), \
         patch('sim2sim.evaluation._hinges',side_effect=AssertionError('disabled hinges executed')):
        report=evaluate_submission(bundle.getvalue(),{'source_dir':str(source),'output_dir':str(tmp_path),
            'width':64,'height':48,'metrics':['appearance'],'probes':[{'body':'invalid_disabled_probe'}],
            'reconstruction':{'enabled':True}})
    assert report['status']=='ok',report
    assert set(report['scores'])=={'reset'}


def test_invalid_selector_fails_before_model_or_filesystem_setup():
    report=evaluate_submission(b'',{'metrics':['not-supported']})
    assert report['status']=='failed' and 'unsupported evaluation metrics' in report['error']


@pytest.mark.parametrize('components', [['chamfer'], ['center_of_mass'], ['chamfer','center_of_mass']])
def test_specific_reconstruction_components_skip_disabled_work(tmp_path, components):
    source=Path(__file__).resolve().parents[1]/'environments/blocks';bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w') as archive: archive.writestr('scene.xml',(source/'scene.xml').read_bytes())
    with patch('sim2sim.primitive_metrics._densities',side_effect=AssertionError('unrequested density work')), \
         patch('sim2sim.evaluation._capture',side_effect=AssertionError('unrequested rendering')):
        report=evaluate_submission(bundle.getvalue(),{'source_dir':str(source),'output_dir':str(tmp_path),
            'width':64,'height':48,'metrics':components,'reconstruction':{'samples':128}})
    assert report['status']=='ok',report
    scores=report['scores']['reconstruction']
    assert 'density' not in scores
    assert ('chamfer_distance_m' in scores)==('chamfer' in components)
    assert ('center_of_mass_error_m' in scores)==('center_of_mass' in components)


def test_explicit_reconstruction_selection_enables_without_source_manifest_opt_in(tmp_path):
    root=Path(__file__).resolve().parents[1];source=root/'environments/blocks';bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w') as archive:archive.writestr('scene.xml',(source/'scene.xml').read_bytes())
    with patch('sim2sim.evaluation._capture',side_effect=AssertionError('disabled appearance/dynamics rendered')):
        report=evaluate_submission(bundle.getvalue(),{'source_dir':str(source),'output_dir':str(tmp_path),
            'width':64,'height':48,'metrics':['reconstruction'],'reconstruction':{'enabled':False,'samples':128}})
    assert report['status']=='ok',report
    assert set(report['scores'])=={'reconstruction'}
    assert report['scores']['reconstruction']['chamfer_status']=='ok'
