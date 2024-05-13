
"""Command line interface for :mod:`MSIregNN`.

Why does this file exist, and why not put this in ``__main__``?
You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m MSIregNN`` python will
  execute``__main__.py`` as a script. That means there won't be any
  ``MSIregNN.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``MSIregNN.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

import click

__all__ = [
    "main",
]


@click.command()
@click.option("--name", required=True, help="The name of the person to say hello to")
def main(name):
    """CLI for MSIregNN."""
    # import inside the CLI to make running the --help command faster
    from .api import hello

    hello(name)

# If you want to have a multi-command CLI, see https://click.palletsprojects.com/en/latest/commands/


if __name__ == "__main__":
    main()