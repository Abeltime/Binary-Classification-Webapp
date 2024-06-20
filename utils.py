# -*- coding: UTF-8 -*-
#
# ProjectName: 二分类模型的网站程序
# Description: 
# Author: 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap


def train_model(train_file="训练集1.xlsx"):
    # 读取数据集
    data1 = pd.read_excel(train_file)

    # 提取特征和标签
    X_train = data1.drop(columns=['cancer'])  # 训练集特征
    y_train = data1['cancer']  # 训练集标签

    # 创建随机森林模型
    rf_model = RandomForestClassifier(random_state=42)

    # 训练模型
    rf_model.fit(X_train, y_train)

    return rf_model


def calculate_shap(model, test):

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test)

    return shap_values


def display_shap(shap_df_neg):
    # 绘制负类别的SHAP值的条形图
    plt.figure(figsize=(10, 6))
    bar_plot_neg = sns.barplot(x='SHAP Value', y='Feature', data=shap_df_neg, color='blue')

    # 设置图形属性
    plt.title('Negative Class')
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature')
    plt.xlim(-0.2, 0.2)  # 根据你的数据范围设置合适的值

    # 将大于0的SHAP值的条形改为红色，小于0的改为蓝色，并在每个条形上标注实际值和SHAP值
    for i in range(shap_df_neg.shape[0]):
        shap_value = shap_df_neg.loc[i, 'SHAP Value']  # 负类别的SHAP值
        if shap_value > 0:
            bar_plot_neg.patches[i].set_color('red')
        elif shap_value < 0:
            bar_plot_neg.patches[i].set_color('blue')
        plt.text(shap_value, i, f'{shap_value:.2f}', va='center', ha='center', color='black', fontsize=12)

    return plt


