import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time
import random

# 讀取資料
data = pd.read_excel('data.xlsx')

# 前置處理：剔除數量為零或負值的交易
data = data[data['QUANTITY'] > 0]

# 資料轉換
transactions = data.groupby('INVOICE_NO')['ITEM_ID'].apply(list).values.tolist()

# 定義函式來進行關聯規則分析並計算時間
def analyze_association_rules(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 使用Apriori演算法
    start_time = time.time()
    frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
    apriori_time = time.time() - start_time
    
    # 使用FP-Growth演算法
    start_time = time.time()
    frequent_itemsets_fpgrowth = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence)
    fpgrowth_time = time.time() - start_time
    
    # 剔除冗餘規則
    rules_apriori = remove_redundant_rules(rules_apriori)
    rules_fpgrowth = remove_redundant_rules(rules_fpgrowth)
    
    return rules_apriori, apriori_time, rules_fpgrowth, fpgrowth_time

# 定義函式來剔除冗餘規則
def remove_redundant_rules(rules):
    redundant_rules = set()
    for idx, rule1 in rules.iterrows():
        for _, rule2 in rules.iterrows():
            # 檢查是否存在 X'⊂X 且 conf(X'→Y) > conf(X→Y)
            if idx != _ and rule2['antecedents'].issubset(rule1['antecedents']) and rule2['confidence'] > rule1['confidence']:
                redundant_rules.add(idx)
    
    # 剔除冗餘規則
    rules = rules.drop(redundant_rules)
    #print("剔除冗餘規則後的結果:")
    #print(rules)
    
    return rules

# 定義函式來輸出規則
def save_association_rules(rules, filename):
    rules.to_csv(filename, index=False)

# 定義函式來讀取規則
def load_association_rules(filename):
    return pd.read_csv(filename)

# 定義函式來進行推薦
def recommend_products(input_products, rules, num_recommendations):
    recommended_products = set()
    for product in input_products:
        
        #print(f"Rules1: {rules}")
        
        # 找到包含輸入產品的規則
        relevant_rules = rules[rules['antecedents'].apply(lambda x: product in set(x))]
        
        # 打印出相關規則和輸入產品
        #print(f"產品: {product}")
        #print(f"相關規則: {relevant_rules}")
        
        # 根據信心度選擇推薦產品
        for _, row in relevant_rules.iterrows():
            recommended_products.update(row['consequents'])
    
    # 隨機從推薦的產品中選擇指定數量
    recommended_products = list(recommended_products)[:num_recommendations]
    
    return recommended_products


# 設定支持度和信心度的值
min_support_values = [0.001, 0.0015, 0.002]
min_confidence_values = [0.1, 0.15, 0.2]

# 記錄結果
results = []

# 進行分析
for min_support in min_support_values:
    for min_confidence in min_confidence_values:
        print(f"使用支持度={min_support}和信心度={min_confidence}進行分析")
        rules_apriori, apriori_time, rules_fpgrowth, fpgrowth_time = analyze_association_rules(
            transactions, min_support, min_confidence)
        
        # 輸出規則
        save_association_rules(rules_apriori, f'rules_apriori_{min_support}_{min_confidence}.csv')
        save_association_rules(rules_fpgrowth, f'rules_fpgrowth_{min_support}_{min_confidence}.csv')

        # 隨機選擇一組產品進行推薦
        # 隨機選擇一筆交易的前三個產品作為輸入
        random_transaction = random.choice(transactions)
        input_products = transactions[random.randint(0, len(transactions) - 1)][:1]

        # 輸出選中的交易和相應的前三個產品
        #print(f"選擇的交易: {random_transaction}")
        #print(f"輸入產品: {input_products}")

        # 檢查選擇的 input_products 是否至少有一個相關的規則
        while all(rules_apriori['antecedents'].apply(lambda x: not any(item in set(x) for item in input_products))):
            #print("選擇的輸入產品沒有找到相關規則。重新選擇一組輸入產品。")
            input_products = transactions[random.randint(0, len(transactions) - 1)][:1]
        #print(f"選擇的交易: {random_transaction}")
        print(f"輸入產品: {input_products}")

        # 推薦
        recommended_products_apriori = recommend_products(input_products, rules_apriori, num_recommendations=5)
        recommended_products_fpgrowth = recommend_products(input_products, rules_fpgrowth, num_recommendations=5)

        
        print(f"Apriori 推薦產品: {recommended_products_apriori}")
        print(f"FP-Growth 推薦產品: {recommended_products_fpgrowth}")
        
        results.append({
            'min_support': min_support,
            'min_confidence': min_confidence,
            'apriori_rules': rules_apriori.shape[0],
            'apriori_time': apriori_time,
            'fpgrowth_rules': rules_fpgrowth.shape[0],
            'fpgrowth_time': fpgrowth_time
        })

# 將結果轉換為DataFrame
results_df = pd.DataFrame(results)

# 儲存結果
results_df.to_csv('association_rules_results.csv', index=False)

# 顯示結果
print(results_df)
