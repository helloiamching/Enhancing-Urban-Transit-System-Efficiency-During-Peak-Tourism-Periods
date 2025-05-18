# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 03:30:04 2025

@author: Nie Haiyi
"""

"""
将月度游客数据转换为日度数据，并考虑地区距离、节假日等因素对游客流量的影响。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays  
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose

# 全局缓存，用于存储主要节假日计算结果，避免重复计算
_MAJOR_HOLIDAYS_CACHE = {}

def read_and_prepare_data(file_path):
    """
    读取原始数据并转换为DataFrame，方便后续处理。
    """
    data = pd.read_csv(file_path)
    
    # 第一列作为时间列，其他列为不同地区来源
    id_var = data.columns[0]
    value_vars = data.columns[1:]
    
    # 将宽表转换成长表，每行对应一个地区在某月的游客数
    data_long = pd.melt(data, id_vars=id_var, value_vars=value_vars, 
                         var_name="Origin", value_name="Visitors")
    
    # 清洗时间字段并转换为 datetime 类型（格式如 "2024 Jan"）
    data_long['Time'] = data_long[id_var].str.strip()
    data_long['Time'] = pd.to_datetime(data_long['Time'], format='%Y %b')
    
    return data_long

def filter_data(data):
    """
    筛选所需记录。
    """
    data_2024 = data[data['Time'].between('2024-01-01', '2024-12-31')]
    data_2025 = data[data['Time'].between('2025-01-01', '2025-01-31')]
    return data_2024, data_2025

def calculate_region_summary(data_2024):
    """
    计算各地区在2024年的平均月度游客数，并计算其占总量的百分比和累积占比。
    将“Other”地区排在最后（不参与前91%主力地区的统计）。
    """
    # 各地区平均月度游客数
    region_summary = data_2024.groupby('Origin')['Visitors'].mean().reset_index()
    region_summary = region_summary.rename(columns={'Visitors': 'Avg_Total'})
    
    total_visitors = region_summary['Avg_Total'].sum()
    region_summary['Percentage'] = 100 * region_summary['Avg_Total'] / total_visitors
    
    # 排序：除 "Other" 外按平均游客数降序
    region_summary['SortOrder'] = region_summary.apply(
        lambda x: -1 if x['Origin'] == 'Other' else x['Avg_Total'], axis=1)
    region_summary = region_summary.sort_values('SortOrder', ascending=False).drop(columns=['SortOrder'])
    
    # 计算累积百分比
    region_summary['Cumulative_Percentage'] = region_summary['Percentage'].cumsum()
    return region_summary

def get_top_regions(region_summary, threshold=91):
    """
    根据累积占比提取主要客源地区列表。
    默认选择累积占比不超过 threshold% 的地区作为重点地区。
    """
    top_regions = region_summary[region_summary['Cumulative_Percentage'] <= threshold]['Origin'].tolist()
    return top_regions

def classify_regions_by_distance():
    """
    按与新加坡的距离将国家/地区分为三类：
    - 近距离：周末短假也可能赴新（1-2天假期足够）
    - 中距离：通常需要3-4天连续假期
    - 远距离：通常需要5天或更长的连续假期
    返回一个字典，键为地区名称，值为距离类别 ('nearby'/'medium'/'long')。
    """
    nearby_regions = ['Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Vietnam']
    medium_distance_regions = [
        'China', 'Hong Kong', 'Taiwan', 'Japan', 'South Korea', 'India', 
        'Australia', 'New Zealand', 'Myanmar', 'Bangladesh'
    ]
    long_distance_regions = [
        'United Kingdom', 'USA', 'Canada', 'Germany', 'France'
    ]
    
    region_distance = {}
    for region in nearby_regions:
        region_distance[region] = 'nearby'
    for region in medium_distance_regions:
        region_distance[region] = 'medium'
    for region in long_distance_regions:
        region_distance[region] = 'long'
    return region_distance

def get_major_holidays(year):
    """
    获取指定年份主要节假日及其权重因子。
    返回一个字典：键为datetime日期，值为(节日名称, 权重)。
    主要包含农历新年、国庆节、劳动节、端午节、中秋节，以及寒暑假等重要假期。
    """
    # 若已计算过该年的主要节假日，则直接返回缓存结果，避免重复计算
    if year in _MAJOR_HOLIDAYS_CACHE:
        return _MAJOR_HOLIDAYS_CACHE[year]
    
    major_holidays = {}
    
    # 使用中国法定节假日来获取农历节日日期（春节、端午、中秋等）
    try:
        cn_holidays = holidays.country_holidays('CN', years=year)
    except Exception as e:
        cn_holidays = {}
    
    # 1. **春节** (农历新年)及扩展：核心7天权重2.5，前后各扩展，外围权重2.0
    cny_main = None
    for date, name in cn_holidays.items():
        # 查找包含 "Spring Festival" 或 "Chinese New Year" 关键字的假日
        if any(keyword in str(name) for keyword in ["Spring Festival", "Chinese New Year", "春节"]):
            # 取最早出现的春节主日期
            if cny_main is None or date < cny_main:
                cny_main = date
    if cny_main is not None:
        # 从春节前2天开始，连续15天作为扩展期（大年初一至正月十五），共计15天
        cny_start = pd.Timestamp(cny_main) - pd.Timedelta(days=2)
        for i in range(15):
            day = cny_start + pd.Timedelta(days=i)
            # 春节核心期（初一到初七）权重2.5，前后边际期权重2.0
            weight = 2.5 if 1 <= i <= 7 else 2.0
            major_holidays[day] = ('Chinese New Year', weight)
    
    # 2. **国庆节** (10月1日起7天长假)：前3天权重2.0，后4天权重1.8
    national_day_start = pd.Timestamp(year=year, month=10, day=1)
    for i in range(7):
        day = national_day_start + pd.Timedelta(days=i)
        weight = 2.0 if i < 3 else 1.8
        major_holidays[day] = ('National Day', weight)
    
    # 3. **劳动节** (5月1日起假期)：持续5天假期，权重1.5
    labor_day_start = pd.Timestamp(year=year, month=5, day=1)
    for i in range(5):
        day = labor_day_start + pd.Timedelta(days=i)
        major_holidays[day] = ('Labor Day', 1.5)
    
    # 4. **端午节** (农历五月初五)：包含节日前后各一天
    dragon_boat_main = None
    for date, name in cn_holidays.items():
        if any(keyword in str(name) for keyword in ["Dragon Boat", "端午"]):
            dragon_boat_main = date
            break
    if dragon_boat_main:
        for i in range(-1, 2):
            day = pd.Timestamp(dragon_boat_main) + pd.Timedelta(days=i)
            major_holidays[day] = ('Dragon Boat Festival', 1.3)
    
    # 5. **中秋节** (农历八月十五)：包含节日前后各一天
    mid_autumn_main = None
    for date, name in cn_holidays.items():
        if any(keyword in str(name) for keyword in ["Mid-Autumn", "中秋"]):
            mid_autumn_main = date
            break
    if mid_autumn_main:
        for i in range(-1, 2):
            day = pd.Timestamp(mid_autumn_main) + pd.Timedelta(days=i)
            major_holidays[day] = ('Mid-Autumn Festival', 1.4)

    
    # 缓存计算结果并返回
    _MAJOR_HOLIDAYS_CACHE[year] = major_holidays
    return major_holidays

def get_effective_holidays(region, year):
    """
    根据地区的距离类别和文化背景，获取该地区在指定年份内 **有效** 的假期日期及其节日信息。
    有效假期指可能显著影响该地区游客出行的假期：
      - 近距离地区：所有周末和法定节假日都视为有效假期（哪怕只有1-2天）。
      - 中距离地区：只有连续3天及以上的假期才视为有效（单独的周末或短假不算）。
      - 远距离地区：只有连续5天及以上的假期才视为有效。
    返回一个字典：键为有效假期日期，值为(假期名称, 假期权重)。
    """
    # 获取地区对应的距离类别，默认为 'long'
    region_distance = classify_regions_by_distance()
    distance_category = region_distance.get(region, 'long')
    
    # 获取主要节假日及权重（如春节、国庆等），用于识别特殊假期和赋予更高权重
    major_holidays = get_major_holidays(year)
    
    # 获取该地区所属国家的法定节假日日期列表
    region_to_country = {
        'China': 'CN', 
        'Indonesia': 'ID', 
        'India': 'IN',
        'Malaysia': 'MY', 
        'Australia': 'AU',
        'Philippines': 'PH', 
        'USA': 'US', 
        'South Korea': 'KR', 
        'United Kingdom': 'GB', 
        'Japan': 'JP',
        'Taiwan': 'TW', 
        'Thailand': 'TH', 
        'Vietnam': 'VN', 
        'Germany': 'DE',
        'Hong Kong': 'HK',
        'France': 'FR',
        'New Zealand': 'NZ',
        'Myanmar': 'MM', 
        'Canada': 'CA',
        'Bangladesh': 'BD'
    }
    country_code = region_to_country.get(region)
    country_holiday_dates = []
    if country_code:
        try:
            # 获取该国家全年法定节假日日期（使用 holidays 库）
            country_holidays = holidays.country_holidays(country_code, years=year)
            country_holiday_dates = list(country_holidays.keys())
        except Exception as e:
            country_holiday_dates = []
    
    # 生成全年所有日期 DataFrame，用于计算连续假期段
    all_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
    date_features = pd.DataFrame({
        'date': all_dates,
        # 周末标记（星期六=5, 星期日=6）
        'is_weekend': all_dates.dayofweek.isin([5, 6]),
        # 法定节假日标记（是否在国家法定假日列表中）
        'is_holiday': all_dates.isin(country_holiday_dates)
    })
    
    # 标记任意休息日（周末或法定假日）
    date_features['is_off'] = date_features['is_weekend'] | date_features['is_holiday']
    # 给每段连续的休息日（周末/假日连续段）分配标识号
    date_features['off_period'] = (date_features['is_off'] != date_features['is_off'].shift()).cumsum()
    # 计算每个连续休息日段的长度（天数）
    date_features['off_period_size'] = date_features.groupby('off_period')['is_off'].transform('sum')
    
    # 根据距离类别设置连续假期的最小天数要求，并标记“有效”的休息日
    if distance_category == 'long':
        min_length = 5
    elif distance_category == 'medium':
        min_length = 3
    else:  # nearby
        min_length = 2
    date_features['is_off_valid'] = date_features['is_off'] & (date_features['off_period_size'] >= min_length)
    
    # 提取有效假期的日期列表
    valid_holiday_dates = date_features[date_features['is_off_valid']]['date'].tolist()
    
    # 构建有效假期信息字典：优先使用 major_holidays 中定义的特殊假期名称及权重
    holiday_info = {}
    for d in valid_holiday_dates:
        if d in major_holidays:
            # 如果是我们定义的重大节假日，则使用对应的名称和权重
            holiday_name, weight = major_holidays[d]
            holiday_info[d] = (holiday_name, weight)
        else:
            # 一般假期（不在重大节假日列表中）统一标记
            holiday_info[d] = ('Regular Holiday', 1.0)
    
    # 根据地区文化调整特定节日的权重：如春节和国庆对特定地区的影响更大
    for d, (name, weight) in list(holiday_info.items()):
        # 如果是假定“春节”并且该地区是受中华文化影响较大的地区，提升权重
        if name == 'Chinese New Year':
            if region in ['China', 'Hong Kong', 'Taiwan', 'Malaysia']:
                holiday_info[d] = (name, weight * 1.5)  # 春节对这些地区影响极大
            elif region in ['Japan', 'Korea', 'Thailand', 'Vietnam', 'Indonesia']:
                holiday_info[d] = (name, weight * 1.2)  # 对这些地区有一定影响
        # 如果是假定“国庆节”并且来源地区是中国，提升权重（中国游客国庆黄金周出行意愿强）
        elif name == 'National Day' and region == 'China':
            holiday_info[d] = (name, weight * 1.3)
        # 如果是假定“暑假”并且地区为东亚国家/地区，略微提升（暑假家庭出游增多）
        elif name == 'Summer Break':
            if region in ['China', 'Hong Kong', 'Taiwan', 'Japan', 'Korea']:
                holiday_info[d] = (name, weight * 1.2)
    
    return holiday_info

def time_series_to_daily_with_distance(region, monthly_data, year):
    """
    将指定地区的月度游客数据转换为日度数据。
    考虑地区距离带来的出行时段偏好，以及重大节假日对游客出行的影响，调整每日游客分布。
    返回包含每日游客流量的 DataFrame，列包括: Time（日期）、Origin（地区）、Visitors（日访客数）、is_holiday（是否假日）、holiday_name（假日名称）。
    """
    # 获取该地区有效假期信息（日期 -> (假期名称, 权重)）
    effective_holidays_info = get_effective_holidays(region, year)
    effective_holidays = set(effective_holidays_info.keys())  # 有效假期日期集
    
    # 获取距离分类
    region_distance = classify_regions_by_distance()
    distance_category = region_distance.get(region, 'long')
    
    # 确保月度数据按时间排序
    monthly_data = monthly_data.sort_values('Time').copy()
    
    # 为缺失的月份填充数据：按月频率重采样并用插值补齐缺失值
    monthly_data_copy = monthly_data.copy()
    monthly_data_copy = monthly_data_copy.set_index('Time').resample('MS').asfreq()  # MS表示每月开始
    monthly_data_copy['Visitors'] = monthly_data_copy['Visitors'].interpolate(method='cubic')
    monthly_data_copy = monthly_data_copy.reset_index()
    
    # 生成全年每日日期框架
    start_date = pd.Timestamp(year=year, month=1, day=1)
    end_date = pd.Timestamp(year=year, month=12, day=31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_data = pd.DataFrame({'Time': date_range})
    daily_data['Origin'] = region  # 标记地区
    
    # 提取日期的时间特征
    daily_data['dayofweek'] = daily_data['Time'].dt.dayofweek      # 星期几（0=周一,...6=周日）
    daily_data['dayofmonth'] = daily_data['Time'].dt.day           # 每月日期
    daily_data['month'] = daily_data['Time'].dt.month              # 月份
    daily_data['quarter'] = daily_data['Time'].dt.quarter          # 季度
    daily_data['is_weekend'] = daily_data['dayofweek'].isin([5, 6]).astype(int)  # 周末标记
    
    # 假日相关信息列初始化
    daily_data['is_effective_holiday'] = daily_data['Time'].isin(effective_holidays).astype(int)
    daily_data['holiday_name'] = None
    daily_data['holiday_weight'] = 1.0  # 默认假日权重为1（普通日）
    # 填充假期名称和权重
    for idx, row in daily_data.iterrows():
        if row['Time'] in effective_holidays_info:
            name, weight = effective_holidays_info[row['Time']]
            daily_data.at[idx, 'holiday_name'] = name
            daily_data.at[idx, 'holiday_weight'] = weight
    
    # 计算假期因子：假期日期使用假期权重，非假期工作日根据距离类别给予不同基准因子
    daily_data['holiday_factor'] = 1.0
    # 假期日的因子 = 专属假期权重
    holiday_mask = (daily_data['is_effective_holiday'] == 1)
    daily_data.loc[holiday_mask, 'holiday_factor'] = daily_data.loc[holiday_mask, 'holiday_weight']
    # 非假期且非周末（普通工作日）的因子，远距离<中距离<近距离
    workday_mask = (daily_data['is_effective_holiday'] == 0) & (daily_data['is_weekend'] == 0)
    if distance_category == 'nearby':
        daily_data.loc[workday_mask, 'holiday_factor'] = 0.85
    elif distance_category == 'medium':
        daily_data.loc[workday_mask, 'holiday_factor'] = 0.80
    else:  # long
        daily_data.loc[workday_mask, 'holiday_factor'] = 0.75
    
    # 标记每条记录所属的月份（用于之后按月汇总）
    daily_data['month_start'] = daily_data['Time'].dt.to_period('M').dt.to_timestamp()
    monthly_data['month_start'] = monthly_data['Time'].dt.to_period('M').dt.to_timestamp()
    
    # 将月度总游客数合并到每日数据上（为每一天附上所在月的总游客数）
    daily_data = pd.merge(daily_data, monthly_data[['month_start', 'Visitors']], 
                           on='month_start', how='left')
    daily_data = daily_data.rename(columns={'Visitors': 'monthly_visitors'})
    
    # 按星期几分配日变化因子（周末效应），根据距离远近设置不同的模式（仅在非节假日应用）
    if distance_category == 'nearby':
        # 近距离：周末效应强，周五开始增加，周六达峰，周日略降
        day_of_week_pattern = {0: 0.7, 1: 0.6, 2: 0.65, 3: 0.75, 4: 1.1, 5: 1.7, 6: 1.5}
    elif distance_category == 'medium':
        # 中距离：周末效应中等，工作日稍低
        day_of_week_pattern = {0: 0.8, 1: 0.75, 2: 0.8, 3: 0.85, 4: 1.1, 5: 1.35, 6: 1.25}
    else:
        # 远距离：周末效应较弱，但周末略高于工作日
        day_of_week_pattern = {0: 0.95, 1: 0.90, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.10, 6: 1.05}
    daily_data['dow_factor'] = daily_data['dayofweek'].map(day_of_week_pattern)
    daily_data.loc[daily_data['is_effective_holiday'] == 1, 'dow_factor'] = 1
    
    # 计算综合因子，使用星期几因子和假日因子
    daily_data['combined_factor'] = daily_data['dow_factor'] * daily_data['holiday_factor']
    
    # 特殊调整：中国游客在春节期间的出行额外增强（在既有holiday_factor基础上进一步放大）
    if region == 'China':
        # 找到标记为春节的日期，额外乘以1.1的因子
        cny_dates = [d for d, (name, _) in effective_holidays_info.items() if name == 'Chinese New Year']
        for d in cny_dates:
            idx = daily_data[daily_data['Time'] == d].index
            if len(idx) > 0:
                daily_data.loc[idx, 'combined_factor'] *= 1.1
    
    # **按月缩放**：将每日相对量调整为匹配实际月度总游客数
    def adjust_to_month_total(group):
        # 以该月 monthly_visitors（月总游客）为基准，按比例分配给每一天
        if 'monthly_visitors' in group.columns and not group['monthly_visitors'].isna().all():
            monthly_total = group['monthly_visitors'].iloc[0]
            sum_factors = group['combined_factor'].sum()
            # 根据组合因子占比分配月总量
            group['adjusted_visitors'] = group['combined_factor'] * monthly_total / (sum_factors if sum_factors != 0 else 1)
        else:
            # 若该月无游客数据，则标记为0访客
            group['adjusted_visitors'] = 0.0
        return group
    daily_data = daily_data.groupby('month_start', group_keys=False).apply(adjust_to_month_total).reset_index(drop=True)
    
    # 整理输出列：日期、地区、日访客数、假期标记、假期名称
    result_data = daily_data[['Time', 'Origin', 'adjusted_visitors', 'is_effective_holiday', 'holiday_name']].copy()
    result_data = result_data.rename(columns={
        'adjusted_visitors': 'Visitors',
        'is_effective_holiday': 'is_holiday'
    })
    return result_data

def process_top_regions(date, top_regions, year):
    """
    针对主要客源地区（top_regions列表中的地区），应用基于距离和假期模型的日度拆分。
    """
    all_adjusted_data = []
    for region in top_regions:
        print(f"处理地区: {region}")
        region_data = date[date['Origin'] == region].copy()

        adjusted_data = time_series_to_daily_with_distance(region, region_data, year)
        all_adjusted_data.append(adjusted_data)
            
    if all_adjusted_data:
        return pd.concat(all_adjusted_data, ignore_index=True)
    else:
        return pd.DataFrame()

def process_remaining_regions(date, region_summary, year, threshold=91):
    """
    针对非主要客源地区（累积占比超过threshold的其余地区），同样应用基于距离的模型进行日度拆分。
    返回这些地区的日度游客数据。
    """
    remaining_regions = region_summary[region_summary['Cumulative_Percentage'] > threshold]['Origin'].tolist()
    if not remaining_regions:
        return pd.DataFrame()
    all_remaining_data = []
    for region in remaining_regions:
        region_data = date[date['Origin'] == region].copy()
        if not region_data.empty:
            adjusted_data = time_series_to_daily_with_distance(region, region_data, year)
            all_remaining_data.append(adjusted_data)
    if all_remaining_data:
        return pd.concat(all_remaining_data, ignore_index=True)
    else:
        return pd.DataFrame()

def main(file_path, output_path, summary_output_path):
    """
    主函数：读取数据，执行数据处理流程，并将结果保存到指定文件。
    参数：
        file_path: 输入月度游客数据文件路径（CSV）。
        output_path: 输出处理后的日度数据文件路径（CSV）。
        summary_output_path: 输出每日游客数汇总表文件路径（CSV）。
    返回：
        final_data, daily_summary 元组，分别为最终日度数据和每日游客数汇总数据（DataFrame）。
    """
    
    # 1. 读取并准备数据
    historical_data = read_and_prepare_data(file_path)
    # 2. 筛选数据
    data_2024, data_2025 = filter_data(historical_data)
    # 3. 计算各地区游客占比摘要
    region_summary = calculate_region_summary(data_2024)
    # 4. 确定主要客源地区列表（累积占比<=91%的地区）
    top_regions = get_top_regions(region_summary, threshold=91)
    # 5. 处理主要地区的日度数据拆分
    adjusted_top_data_2024 = process_top_regions(data_2024, top_regions, year=2024)
    adjusted_top_data_2025 = process_top_regions(data_2025, top_regions, year=2025)
    # 6. 处理其余地区的日度数据拆分
    adjusted_remaining_data_2024 = process_remaining_regions(data_2024, region_summary, year=2024, threshold=91)
    adjusted_remaining_data_2025 = process_remaining_regions(data_2025, region_summary, year=2025, threshold=91)
    # 7. 合并所有地区的数据
    final_data = pd.concat([adjusted_top_data_2024, adjusted_remaining_data_2024, adjusted_top_data_2025, adjusted_remaining_data_2025], ignore_index=True)
    final_data["Visitors"] = final_data["Visitors"].astype(int)
    
    # 8. 删除visitor为0的数据
    final_data = final_data[final_data["Visitors"] > 0]
    
    # 9. 按照国家和时间排序
    final_data = final_data.sort_values(by=["Origin", "Time"])
    
    # 10. 保存详细日度数据到 CSV
    final_data.to_csv(output_path, index=False)
    
    # 11. 创建每日游客数汇总表（所有国家的数据按日期汇总）
    daily_summary = final_data.groupby("Time")["Visitors"].sum().reset_index()
    daily_summary.columns = ["Time", "Total_Visitors"]
    daily_summary = daily_summary.sort_values(by="Time")
    
    # 12. 保存每日游客数汇总表到 CSV
    daily_summary.to_csv(summary_output_path, index=False)
    
    return final_data, daily_summary

if __name__ == "__main__":
    input_file = "../data/process_travel_arrivals.csv"  # 输入月度数据文件
    output_file = "../data/final_daily_tourists.csv"    # 输出日度数据文件
    summary_file = "../data/daily_summary_tourists.csv" # 输出每日游客数汇总表文件
    
    final_data, daily_summary = main(input_file, output_file, summary_file)
