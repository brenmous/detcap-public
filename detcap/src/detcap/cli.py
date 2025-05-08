"""
The Detcap CLI. If CLI options are not set, they are derived from 
environment variables which have default fallbacks. See 
:mod:`~detcap.settings` for the definition of these env vars.

.. click:: detcap.cli:cli
    :prog: detcap
    :nested: full
"""
import asyncio
from contextlib import nullcontext

import click

from detcap import (
    configure_logger,
    configure_exception_handler,
    settings,
    app,
)


# If CLI options are not set, they are derived from environment variables which have default
# fallbacks. See settings.py for the definition of these env vars.
@click.group()
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default=settings.LOG_LEVEL,
    help="Logging level",
)
@click.option(
    "-l",
    "--log-file",
    type=click.Path(dir_okay=False),
    default=settings.LOG_FILE,
    help="File to save log output to. If not provided, logging is limited to console. "
    "It's recommended to use this rather than output redirection, as this uses "
    "a rotating file handler to prevent the log file continuously growing",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=settings.LOG_QUIET,
    help="Silences console logging output",
)
@click.option(
    "-n",
    "--name-tag",
    type=str,
    required=False,
    help=(
        "An optional name tag. Added to metadata and filename when uploading to SKIP."
    ),
)
@click.pass_context
def cli(ctx, verbosity, log_file, quiet, name_tag):
    """ """
    configure_logger(verbosity, log_file, quiet)
    configure_exception_handler()
    ctx.obj = {'nametag': name_tag}



@cli.command()
@click.argument(
    "config-file",
    type=click.Path(dir_okay=False, exists=True),
    default=settings.CONFIG_FILE,
)
@click.option(
    "-s",
    "--seedlink-server",
    type=str,
    default=settings.SEEDLINK_SERVER,
    help="URL of the Seedlink server. This is without protocol, e.g. 'localhost:18000'",
)
@click.option(
    "-i",
    "--inventory",
    type=click.Path(dir_okay=False),
    required=False,
    default=settings.INVENTORY,
    help="Inventory to use, StationXML format. If not provided, then the"
    "inventory is fetched from FDSN. If provided and it doesn't yet "
    "exist, the inventory is fetched from FDSN and saved to this file.",
)
@click.option(
    "-f",
    "--fdsn",
    type=str,
    required=False,
    default=settings.FDSN_SERVER,
    help="FDSN URL, used to fetch inventory if no file is provided. This is with "
    "protocol, e.g. 'http://localhost:8081'",
)
@click.option(
    "-d",
    "--save-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    required=False,
    default=settings.SAVE_DIRECTORY,
    help="Directory to save map files to. If not provided, map files are stored temporarily.",
)
@click.pass_context
def monitor(
    ctx,
    config_file,
    seedlink_server,
    inventory,
    fdsn,
    save_dir,
):
    """
    Begin realtime monitoring using the regions and parameters defined
    in a Detcap config file.
    """
    application = app.DetcapApp(
        config_file,
        seedlink_server,
        inventory,
        fdsn,
        save_dir,
        ctx.obj['nametag'],
    )
    asyncio.run(application())


if __name__ == "__main__":
    cli()
