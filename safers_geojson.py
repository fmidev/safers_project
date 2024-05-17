"""
SAFERS geoJSON utilities.
"""

# marko.laine@fmi.fi

from os.path import exists

import numpy as np
from osgeo import ogr, osr
from matplotlib.pyplot import contourf, ioff

# https://gdal.org/drivers/vector/geojson.html
geoJSON_Options = ["SIGNIFICANT_FIGURES=4", "COORDINATE_PRECISION=4", "WRITE_BBOX=YES"]


#
# Convert data array to geojson isobands
#
# The contour to isobands code has been adapted from
# https://github.com/rveciana/geoexamples/tree/master/python/raster_isobands
#
def da_to_geojson(da, filename, interval='auto',
                  layer_name='forecast_field',
                  out_format='GeoJSON',
                  nbands=20,
                  epgs=4326, x='lon', y='lat'):
    """Convert xarray.DataArray to geoJSON isobands."""
    attr_name = da.name
    if len(attr_name) == 0:
        attr_name = 'unknown'

    if interval == 'auto':
        interval = (da.max().values - da.min().values) / nbands
    if interval <= 0.0:
        raise ValueError('Isobar interval is not positive')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epgs)
    drv = ogr.GetDriverByName(out_format)

    if exists(filename):
        drv.DeleteDataSource(filename)
    dst_ds = drv.CreateDataSource(filename)
    dst_layer = dst_ds.CreateLayer(layer_name, geom_type=ogr.wkbPolygon, srs=srs,
                                   options=geoJSON_Options)
    fdef = ogr.FieldDefn(attr_name, ogr.OFTReal)
    dst_layer.CreateField(fdef)

    levels = np.arange(interval * np.floor(da.min() / interval),
                       interval * (1 + np.ceil(da.max() / interval)),
                       interval)
    with ioff():
        contours = contourf(da[x], da[y], da, levels)

    for level in range(len(contours.collections)):
        paths = contours.collections[level].get_paths()
        for path in paths:
            feat_out = ogr.Feature(dst_layer.GetLayerDefn())
            feat_out.SetField(attr_name, contours.levels[level])
            pol = ogr.Geometry(ogr.wkbPolygon)
            ring = None
            for i in range(len(path.vertices)):
                point = path.vertices[i]
                if path.codes[i] == 1:
                    if ring is not None:
                        pol.AddGeometry(ring)
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint_2D(point[0], point[1])
            pol.AddGeometry(ring)
            feat_out.SetGeometry(pol)
            if dst_layer.CreateFeature(feat_out) != 0:
                raise IOError("Problems creating output file")
            feat_out.Destroy()
