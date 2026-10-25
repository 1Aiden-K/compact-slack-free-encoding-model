"""Lazy command dispatch for supported tiers."""
import argparse
import importlib
import sys

COMMANDS = {
    'generate': ('instances', 'generate'),
    'validate': ('instances', 'validate'),
    'benchmark': ('benchmark', None),
    'run': ('run', None),
    'optimize': ('optimization', None),
    'evaluate': ('evaluation', None),
    'classical': ('classical', None),
    'lambda-benchmark': ('lambda_benchmark', None),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Track A capacity-constrained clustering')
    parser.add_argument('--tier', choices=('s', 'm'), default='s', help='instance tier (default: s)')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in COMMANDS:
        commands.add_parser(name, add_help=False, help={
            'generate': 'create certified instances', 'validate': 'check instance certificates',
            'benchmark': 'compare methods across instances', 'run': 'embed, optimize, and emulate one instance',
            'optimize': 'compare original and selected pulses', 'evaluate': 'evaluate saved counts or run',
            'classical': 'inspect the exact constrained and QUBO references',
            'lambda-benchmark': 'compare Tier M fixed and adaptive penalty ground states',
        }[name])
    args, forwarded = parser.parse_known_args(argv)
    if args.tier == 's' and args.command == 'lambda-benchmark':
        parser.error('lambda-benchmark is only available for Tier M.')
    if args.tier == 'm' and args.command not in ('generate', 'validate', 'classical', 'evaluate', 'lambda-benchmark'):
        parser.error('Tier M currently supports generate, validate, classical, and evaluate. '
                     'Quantum execution and pulse optimization are not integrated yet.')
    module_name, prefix = COMMANDS[args.command]
    module = importlib.import_module(f'duquantum.tier_{args.tier}.{module_name}')
    old_argv = sys.argv
    try:
        sys.argv = [f'main.py {args.command}', *([prefix] if prefix else []), *forwarded]
        result = module.main()
        return 0 if result is None else result
    finally:
        sys.argv = old_argv
