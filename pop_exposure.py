import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask

# -------------------------
# 文件路径
# -------------------------
districts_shp = "nanjing_detail.shp"
pop_raster_file = "pop_mask.tif"
flood_raster_file = "flood_100.tif"
output_csv = "nanjing_flood_exposure_by_district.csv"

# -------------------------
# ① 读取区县 shapefile
# -------------------------
districts = gpd.read_file(districts_shp)
district_field = 'name'  # 区县名称字段
print("区县字段:", district_field)

# -------------------------
# ② 读取栅格
# -------------------------
with rasterio.open(pop_raster_file) as pop_src:
    pop_data = pop_src.read(1).astype(np.float64)
    pop_transform = pop_src.transform
    pop_crs = pop_src.crs
    pop_nodata = pop_src.nodata
    if pop_nodata is not None:
        pop_data[pop_data == pop_nodata] = 0

with rasterio.open(flood_raster_file) as flood_src:
    flood_data = flood_src.read(1).astype(np.float64)
    flood_transform = flood_src.transform
    flood_crs = flood_src.crs
    flood_nodata = flood_src.nodata
    if flood_nodata is not None:
        flood_data[flood_data == flood_nodata] = 0

# -------------------------
# ③ CRS 对齐
# -------------------------
if districts.crs != pop_crs:
    districts = districts.to_crs(pop_crs)

# -------------------------
# ④ 计算总人口和暴露人口
# -------------------------
results = []

for idx, row in districts.iterrows():
    geom = [row.geometry]

    # 生成区县掩膜
    mask = geometry_mask(geom, transform=pop_transform, invert=True, out_shape=pop_data.shape)

    # 区县总人口
    total_pop = np.sum(pop_data[mask])

    # 区县暴露人口（人口栅格 * 洪涝栅格）
    exposed_pop = np.sum(pop_data[mask] * flood_data[mask])

    # 暴露率
    exposure_rate = exposed_pop / total_pop if total_pop > 0 else 0

    results.append({
        'district': row[district_field],
        'total_pop': total_pop,
        'exposed_pop': exposed_pop,
        'exposure_rate': exposure_rate
    })

# -------------------------
# ⑤ 输出 CSV
# -------------------------
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(df)
