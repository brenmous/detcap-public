https://brenmous.github.io/detcap-public

Installation is thoroughly outdated but the implementation section is still
relevant.

Rough guide to relevant code:
- `app.py` has an update function, this triggers every e.g. 60 seconds
  and starts the processing
- `inventory/station.py` has the `handle_traces` function which preprocesses
  the traces (filtering, seismometer simulation)
- `product.update` is then called, where each "product" is a map/grid defined
  by the config
- A base class for the map product is in `maps/__init__.py`. This has a main
  `update` function which orchestrates the calculation, with some function
  implementations handled by `maps/mb.py` and `maps/mla.py` depending on
  selected magnitude.
- The grid is then output as geojson which would be rendered in the front end
