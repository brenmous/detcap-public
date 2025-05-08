"""
Uploading Detcap data to SKIP.
"""
import os
from datetime import datetime
from functools import lru_cache
from typing import Sequence, Optional, Dict, Any
from logging import getLogger
import json

from eatws_skip_client import SKIP, oo as SkipTypes

from detcap import Float32Encoder


_logger = getLogger(__name__)


@lru_cache(maxsize=None)
def skip_client():
    return SKIP(cache=False)


def upload_detcap_product(
    product,
    map_data,
    name_tag: Optional[str] = None,
    logfile=None,
) -> SkipTypes.Product:
    # Probably a silly way of doing this.
    # We need to convert the metadata occurences of float32 to float
    # so they work with skip client's json dump.
    # Rather than recursing the dictionary ourselves, we can use our
    # encoder to dump it converted, then load it back.
    converted_metadata = json.loads(json.dumps(product.metadata, cls=Float32Encoder))
    _logger.debug("Uploading %s to SKIP", product)
    title = f"Detcap: {product.name}"
    title += f" ({name_tag})" if name_tag else ""
    return upload_product(
        product_name=os.path.basename(product.map_file),
        product_type="detcap_map",
        title=title,
        metadata=converted_metadata,
        content=json.dumps(map_data, cls=Float32Encoder).encode("utf-8"),
        mimetype="application/geo+json",
        time=product.last_updated,
        logfile=logfile,
    )


def upload_product(
    product_name: str,
    product_type: str,
    title: str,
    parents: Sequence[SkipTypes.Product] = [],
    metadata: Dict[Any, Any] = {},
    content: Optional[bytes] = None,
    mimetype: Optional[str] = None,
    time: Optional[datetime] = None,
    geography: Optional[str] = None,
    logfile: Optional[str] = None,
    public: bool = False,
) -> SkipTypes.Product:
    """
    Uploads a new SKIP product.

    Parameters
    ----------
    product_name
        Name of the product (equivalent to file name).
    product_type
        Product type (SKIP convention is snake_case).
    title
        Human-friendly title of the product.
    parents
        A collection of parent products to associate with this
        product.
    metadata
        Product metadata.
    content
        Content to upload.
    mimetype
        Mimetype of the content.
    time
        A time to associate with this product.
    geography
        Optional geography to associate with the produt as a
        WKT string.
    logger
        A logger to use during upload. If FileHandlers are attached,
        the first FileHandler found will be uploaded to SKIP as the
        creation log (i.e. pass in the logger that has been used
        in the creation steps of the product).

    Returns
    -------
    SkipTypes.product
        The uploaded product.
    """
    data = {
        "event_id": None,
        "product_name": product_name,
        "product_type": product_type,
        "title": title,
        "time": time.isoformat() if time is not None else None,
        "content": content,
        "mimetype": mimetype,
        "geography": geography,
        "parents": parents,
        "metadata": metadata,
        "logfile": logfile,
        "public": public,
    }
    return skip_client().upload_product(**data)
