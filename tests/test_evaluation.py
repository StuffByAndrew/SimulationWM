import base64
import io
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import pytest
from sim2sim.evaluation import decode_submission, extract_submission, evaluate_submission


def archive(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, content in files.items():
            z.writestr(name, content)
    return out.getvalue()


def test_valid_project(tmp_path):
    data = archive({'scene.xml': '<mujoco><worldbody/></mujoco>'})
    assert decode_submission(base64.b64encode(data).decode()) == data
    assert extract_submission(data, tmp_path).is_file()


@pytest.mark.parametrize('files', [
    {'../scene.xml': '<mujoco/>'},
    {'scene.xml': '<mujoco><include file="/etc/passwd"/></mujoco>'},
    {'scene.xml': '<mujoco><extension><plugin plugin="evil"/></extension></mujoco>'},
    {'scene.xml': '<mujoco><include file="part.txt"/></mujoco>', 'part.txt': '<mujoco><include file="/etc/passwd"/></mujoco>'},
    {'scene.xml': '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><mujoco/>'},
    {'scene.xml': '<mujoco><compiler meshdir="../private"/></mujoco>'},
])
def test_reject_unsafe(tmp_path, files):
    with pytest.raises(ValueError):
        extract_submission(archive(files), tmp_path)


def test_symlink(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as z:
        info = zipfile.ZipInfo('scene.xml')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        z.writestr(info, '/etc/passwd')
    with pytest.raises(ValueError):
        extract_submission(output.getvalue(), tmp_path)


def test_bad_submission_explicit_failure(tmp_path):
    report = evaluate_submission(b'not a zip', {'output_dir': str(tmp_path)})
    assert report['status'] == 'failed'
    assert report['scores'] is None


def test_base64_bounds():
    with pytest.raises(ValueError):
        decode_submission('@@')


@pytest.mark.parametrize('files', [
    {'scene.xml': '<mujoco><include file="a.part"/></mujoco>', 'a.part': '<mujoco><include file="b.part"/></mujoco>', 'b.part': '<mujoco><asset><mesh file="../../escape.obj"/></asset></mujoco>'},
    {'scene.xml': '<mujoco><include file="missing.part"/></mujoco>'},
    {'scene.xml': '<mujoco><asset><texture file="https://example.com/image.png"/></asset></mujoco>'},
    {'scene.xml': '<mujoco><include file="bad.part"/></mujoco>', 'bad.part': '<mujoco>'},
])
def test_malformed_or_external_references(tmp_path, files):
    with pytest.raises((ValueError, OSError, ET.ParseError)):
        extract_submission(archive(files), tmp_path)
