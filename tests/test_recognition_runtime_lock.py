"""The vision dependency override must not affect TTS or the parent process."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from components.vision_yolov8_objdetect import process


class RecognitionRuntimeLockTests(unittest.TestCase):
    def test_other_hosts_and_unconfigured_runtimes_keep_original_environment(self):
        with patch.object(process.platform, 'machine', return_value='x86_64'), \
                patch.object(process.subprocess, 'run') as run:
            self.assertIsNone(process.locked_recognition_environment({'runtime': {'use_robot_runtime_lock': True}}))
            self.assertIsNone(process.locked_recognition_environment({}))
            run.assert_not_called()

    def test_native_overrides_are_child_only_and_use_selected_demo_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root/'backend/components/robot_arm_nero/config.json'
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({'demo_root': '../selected-demo'}))
            source = root/'backend/components/vision_yolov8_objdetect/process.py'
            parent = dict(os.environ)
            with patch.object(process, '__file__', str(source)), \
                    patch.object(process.platform, 'machine', return_value='riscv64'), \
                    patch.object(process.subprocess, 'run') as run:
                run.return_value.stdout = json.dumps({'LD_LIBRARY_PATH': '/locked/lib', 'LD_PRELOAD': '/locked/tcm.so'})
                result = process.locked_recognition_environment({'runtime': {'use_robot_runtime_lock': True}})
                self.assertEqual(result['LD_LIBRARY_PATH'], '/locked/lib')
                self.assertEqual(dict(os.environ), parent)
                self.assertEqual(run.call_args.args[0][1], str(root.parent/'selected-demo/scripts/k3_runtime.py'))

    def test_missing_lock_is_not_silently_ignored(self):
        with patch.object(process.platform, 'machine', return_value='riscv64'), \
                patch.object(process.Path, 'read_text', return_value='{"demo_root":"dice_demo"}'), \
                patch.object(process.subprocess, 'run', side_effect=FileNotFoundError('missing lock')):
            with self.assertRaises(FileNotFoundError):
                process.locked_recognition_environment({'runtime': {'use_robot_runtime_lock': True}})
