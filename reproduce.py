"""Run the supplied analyses from the repository root."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'figures': ('figures', 'draw functional panels from locally fitted results'),
    'fit': ('analysis', 'fit weighted GAMs to the five simulation inputs'),
    'simulate': ('simulation', 'run a small check or a full simulation grid'),
    'graphs': ('graph_construction', 'convert separately obtained biological source tables'),
    'summary': ('network_statistics', 'summarize the locally prepared connectome graphs'),
    'degree-fits': ('distribution_fitting', 'fit five degree-distribution families'),
    'thresholds': ('threshold_sensitivity', 'run synaptic-weight threshold analyses'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use COMMAND --help for command-specific options.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=COMMANDS, help='\n'.join(f'{k}: {v[1]}' for k,v in COMMANDS.items()))
    parser.add_argument('args', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.command in {'summary', 'thresholds'}:
        fixed = argparse.ArgumentParser(prog=f'{Path(sys.argv[0]).name} {options.command}',
                                        description=COMMANDS[options.command][1])
        fixed.parse_args(options.args)
    module = COMMANDS[options.command][0]
    sys.path.insert(0, str(ROOT/'code/src'))
    sys.argv = [f'{options.command}', *options.args]
    runpy.run_module(f'connectome_dynamics.{module}', run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
