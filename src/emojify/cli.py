"""Click CLI definitions for emojify."""

import click

from emojify import __version__


@click.group()
def cli():
    """Bidirectional emoji-text translator."""
    pass


@cli.command()
def version():
    """Print the emojify version."""
    click.echo(f"emojify v{__version__}")
