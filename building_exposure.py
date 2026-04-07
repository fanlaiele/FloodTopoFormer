import os
import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
import pandas as pd

# =====================================================
# 1. 路径设置 (请根据你的实际路径调整)
# =====================================================
bld_path = r"./建筑物/precise_corrected_buildings.shp"
district_path = r"./南京shp/nanjing_detail.shp"
flood_path = r"./flood.tif"

# 最终输出的两个数据表
out_district_csv = "district_flood_stats0324.csv"
out_matrix_csv = "district_height_matrix0324.csv"
# 可选：输出带被淹属性的矢量，方便你在GIS里复查
out_building_shp = "buffered_building_result0324.shp"

BUFFER_DISTANCE = 10  # 【核心修改】：缓冲区距离（单位：米）。代表只要建筑周边10米内有水，就算被淹。刚好分辨率是10m
height_field = "corrected_"

# =====================================================
# 2. 加载数据并处理投影 (极其重要)
# =====================================================
print("Loading data...")
bld = gpd.read_file(bld_path)
districts = gpd.read_file(district_path)

flood_src = rasterio.open(flood_path)
flood_arr = flood_src.read(1)
flood_mask = (flood_arr > 0).astype(np.uint8) # 1 为水，0 为非水
transform = flood_src.transform
crs = flood_src.crs

# 确保所有矢量数据的坐标系与栅格一致
if bld.crs != crs:
    print("Reprojecting buildings to match raster CRS...")
    bld = bld.to_crs(crs)
if districts.crs != crs:
    print("Reprojecting districts to match raster CRS...")
    districts = districts.to_crs(crs)

# 【安全检查】：确保使用的是投影坐标系（单位是米），否则 Buffer(15) 会变成 15度！
if not bld.crs.is_projected:
    print("WARNING: CRS is geographic (degrees). Temporarily reprojecting to Web Mercator (EPSG:3857) for buffering in meters...")
    original_crs = bld.crs
    bld = bld.to_crs("EPSG:3857")
    districts = districts.to_crs("EPSG:3857")
else:
    original_crs = bld.crs

# =====================================================
# 3. 生成缓冲区并进行空间栅格化匹配 (核心算法更新)
# =====================================================
print(f"Creating {BUFFER_DISTANCE}m buffers around buildings...")
bld = bld.reset_index(drop=True)
bld["b_id"] = np.arange(1, len(bld) + 1).astype(np.int32)

# 【核心】：生成带缓冲区的建筑几何体系
buffered_geometry = bld.geometry.buffer(BUFFER_DISTANCE)

# 如果刚才转换了坐标系，现在转回栅格坐标系以便进行 rasterize
if bld.crs != crs:
    buffered_geometry = buffered_geometry.to_crs(crs)
    bld = bld.to_crs(crs)
    districts = districts.to_crs(crs)

print("Rasterizing buffered buildings (this may take a minute)...")
shapes = ((geom, bid) for geom, bid in zip(buffered_geometry, bld["b_id"]))

bld_raster = rasterio.features.rasterize(
    shapes,
    out_shape=flood_mask.shape,
    transform=transform,
    fill=0,
    dtype=np.int32
)

# =====================================================
# 4. 矩阵化极速统计暴露度
# =====================================================
print("Computing flood intersections...")
flat_bld = bld_raster.ravel()
flat_flood = flood_mask.ravel()

mask = flat_bld > 0
bld_ids = flat_bld[mask]
flood_flags = flat_flood[mask]

# 统计每个建筑(含缓冲区)范围内包含的洪水像元数
flood_pix_counts = np.bincount(bld_ids, weights=flood_flags)

# 确保数组长度能覆盖所有 ID
max_id = bld["b_id"].max()
if len(flood_pix_counts) <= max_id:
    flood_pix_counts = np.pad(flood_pix_counts, (0, max_id - len(flood_pix_counts) + 1))

# 【新判定标准】：只要建筑及其周边(15m内)有1个以上的洪水像元，即视为暴露/被淹！
bld["flooded"] = (bld["b_id"].apply(lambda x: flood_pix_counts[x]) > 0).astype(int)

# =====================================================
# 5. 属性处理：高度分级与行政区匹配
# =====================================================
print("Classifying building heights...")
bld[height_field] = pd.to_numeric(bld[height_field], errors="coerce").fillna(0)

def classify_height(h):
    if h <= 10: return "Low (<=10 m)"
    elif h <= 24: return "Mid (10-24 m)"
    elif h <= 60: return "High (24-60 m)"
    else: return "Very High (>60 m)"

bld["height_group"] = bld[height_field].apply(classify_height)

print("Performing spatial join with districts...")
# 自动寻找行政区名称字段
name_cols = [c for c in districts.columns if "name" in c.lower() or "名称" in c]
dist_col = name_cols[0] if name_cols else "district_name"
if dist_col not in districts.columns:
    districts = districts.reset_index().rename(columns={"index": "district_name"})
    dist_col = "district_name"

# 获取每栋建筑对应的行政区
bld_join = gpd.sjoin(bld, districts[[dist_col, "geometry"]], how="left", predicate="intersects")
bld_join[dist_col] = bld_join[dist_col].fillna("Unknown")

# =====================================================
# 6. 生成所需的两个最终 CSV 表格
# =====================================================
print("Aggregating statistics...")

# 表1：行政区整体受灾统计 (district_flood_stats.csv)
district_stats = bld_join.groupby(dist_col).agg(
    total_buildings=("b_id", "count"),
    flooded_buildings=("flooded", "sum")
).reset_index()
district_stats["flood_rate"] = district_stats["flooded_buildings"] / district_stats["total_buildings"]

district_stats.to_csv(out_district_csv, index=False, encoding="utf-8-sig")
print(f"✅ Created: {out_district_csv}")

# 表2：行政区 × 高度详细统计表 (输出为你截图里的格式)
matrix_stats = bld_join.groupby([dist_col, "height_group"]).agg(
    total_buildings=("b_id", "count"),
    flooded_buildings=("flooded", "sum")
).reset_index()
matrix_stats["flood_rate"] = matrix_stats["flooded_buildings"] / matrix_stats["total_buildings"]

# 【修改点】：直接保存这个包含完整数据的长表，不再进行Pivot转换
matrix_stats.to_csv(out_matrix_csv, index=False, encoding="utf-8-sig")
print(f"✅ Created detailed matrix: {out_matrix_csv}")


# =====================================================
# 7. 输出最终的建筑物 Shapefile (原始轮廓 + 淹没属性)
# =====================================================
print("Saving building shapefile (original footprints with flood labels)...")

# 定义最终输出的文件名 (名字你可以随便改)
out_building_shp = "original_buildings_with_flood_label.shp"

# 只保留你需要的核心字段，剔除多余字段防止报错
cols_to_keep = ['b_id', height_field, 'height_group', 'flooded', 'geometry']
bld_out = bld_join[cols_to_keep].copy()

# 保存为 Shapefile
bld_out.to_file(out_building_shp, encoding="utf-8")
print(f"✅ 成功生成 Shapefile: {out_building_shp}")
