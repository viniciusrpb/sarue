import geopandas as gpd

gdf = gpd.read_file("samples/DF_setores_CD2022/DF_setores_CD2022.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("setoresDF.json", driver="GeoJSON")
