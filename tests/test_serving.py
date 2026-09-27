"""Server ownership and occupied-port checks without launching a GPU process."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from cocast.serving import serve, check_server

SETTINGS = {'base_url':'http://127.0.0.1:8000/v1','model':'fixture','model_revision':'a'*40,'context_window':8192}


class ServingTests(unittest.TestCase):
    def test_occupied_port_does_not_download_or_stop_another_server(self):
        connection = MagicMock()
        connection.__enter__.return_value.connect_ex.return_value = 0
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda:1))
        with patch.dict('sys.modules', {'torch':torch}), \
             patch('cocast.serving.socket.socket', return_value=connection), \
             patch('huggingface_hub.snapshot_download') as download, \
             patch('cocast.serving.subprocess.Popen') as process, \
             patch('cocast.serving.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                with serve(SETTINGS, [], Path('/unused')):
                    pass
        download.assert_not_called()
        process.assert_not_called()
        kill.assert_not_called()

    def test_owned_process_group_is_cleaned_up_on_failure(self):
        connection = MagicMock()
        connection.__enter__.return_value.connect_ex.return_value = 1
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda:1))
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict('sys.modules', {'torch':torch}), \
             patch('cocast.serving.socket.socket', return_value=connection), \
             patch('huggingface_hub.snapshot_download', return_value='checkpoint'), \
             patch('cocast.serving.check_server'), \
             patch('cocast.serving.subprocess.Popen', return_value=process) as launch, \
             patch('cocast.serving.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with serve(SETTINGS, [], Path(temp)/'server.log'):
                    raise RuntimeError('fixture')
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual([c.args[0] for c in kill.call_args_list], [12345,12345])
