"""Compile the real divider functions against OpenCV; no model or camera needed."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from components.vision_yolov8_objdetect.profile import load_runtime_config, ProfileError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('colors', ['red_blue', 'green_gray'])
def test_runtime_accepts_explicit_table_colors(tmp_path, capsys, colors):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps({'divider_colors': colors}))
    assert load_runtime_config(path)['divider_colors'] == colors
    assert 'unknown keys' not in capsys.readouterr().out


@pytest.mark.parametrize('colors', ['green', '', None, [], True])
def test_runtime_rejects_invalid_table_colors(tmp_path, colors):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps({'divider_colors': colors}))
    with pytest.raises(ProfileError, match='divider_colors'):
        load_runtime_config(path)


def test_real_divider_handles_both_mats_and_rotations(tmp_path):
    if not shutil.which('g++') or not shutil.which('pkg-config'):
        pytest.skip('OpenCV C++ development tools required')
    flags = subprocess.run(['pkg-config', '--cflags', '--libs', 'opencv4'],
                           capture_output=True, text=True)
    if flags.returncode:
        pytest.skip('OpenCV C++ development files required')
    source = (ROOT / 'vision/yolov8_objdetect/src/main.cpp').read_text()
    functions = source[source.index('static bool detect_black_divider('):
                       source.index('static void draw_scene_assist(')]
    header = '''
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
struct DividerLine {
    bool valid = false, horizontal = false;
    cv::Point2f point, direction, normal;
};
'''
    checks = '''
int main() {
    for (const bool green : {false, true}) {
        for (const bool reverse : {false, true}) {
            for (const bool horizontal : {false, true}) {
                for (const double exposure : {0.65, 1.0}) {
                    const cv::Scalar a = green ? cv::Scalar(55,226,176) : cv::Scalar(20,30,200);
                    const cv::Scalar b = green ? cv::Scalar(102,102,102) : cv::Scalar(220,60,20);
                    cv::Mat mat(480,640,CV_8UC3,reverse ? b : a);
                    mat(cv::Rect(300,0,340,480)).setTo(reverse ? a : b);
                    cv::circle(mat,{150,270},45,cv::Scalar(250,250,250),-1);
                    cv::circle(mat,{480,170},35,cv::Scalar(15,15,15),-1);
                    mat.convertTo(mat,CV_8UC3,exposure);
                    if (horizontal) cv::transpose(mat,mat);
                    DividerLine line;
                    if (!detect_scene_divider(mat,line,green ? "green_gray" : "red_blue") ||
                        line.horizontal != horizontal ||
                        std::abs((horizontal ? line.point.y : line.point.x)-300) > 5) return 1;
                }
            }
        }
    }
    for (const cv::Scalar color : {cv::Scalar(55,226,176),cv::Scalar(102,102,102),cv::Scalar(255,255,255)}) {
        cv::Mat blank(480,640,CV_8UC3,color);
        DividerLine line;
        if (detect_scene_divider(blank,line,"green_gray")) return 2;
    }
    // A small green cup on a gray table must not become a table boundary.
    cv::Mat cup(480,640,CV_8UC3,cv::Scalar(102,102,102));
    cv::circle(cup,{290,240},40,cv::Scalar(55,226,176),-1);
    DividerLine line;
    if (detect_scene_divider(cup,line,"green_gray")) return 3;
    return 0;
}
'''
    cpp = tmp_path / 'divider.cpp'
    binary = tmp_path / 'divider'
    cpp.write_text(header + functions + checks)
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic',
                    str(cpp), '-o', str(binary), *flags.stdout.split()], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=20)
