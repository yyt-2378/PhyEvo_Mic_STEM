"""Sandbox-local library configuration; no historical replay dependency."""
from pathlib import Path

from .contracts import file_hash, inside, write_new
from .sandbox import Sandbox


class ConfiguredSandbox(Sandbox):
    configuration = (('XDG_CONFIG_HOME', '/work/config'),
                     ('MPLCONFIGDIR', '/work/config/matplotlib'),
                     ('MPLBACKEND', 'Agg'))

    def command(self, program, inputs, output, worker):
        command = super().command(program, inputs, output, worker)
        executable = ['/runtime/bin/python', '-I', '-B', '/worker.py']
        if command[-4:] != executable:
            raise ValueError('Sandbox launcher changed; review adapter before use')
        settings = [token for key, value in self.configuration
                    for token in ('--setenv', key, value)]
        return command[:-4] + settings + executable

    def run(self, program, inputs, output):
        receipt = super().run(program, inputs, output)
        info = {'adapter_sha256': file_hash(Path(__file__)),
                'environment': dict(self.configuration),
                'host_mounts_or_permissions_extended': False}
        write_new(inside(self.workspace, output) / 'runtime_configuration.json', info)
        return receipt | {'runtime_configuration': info}
