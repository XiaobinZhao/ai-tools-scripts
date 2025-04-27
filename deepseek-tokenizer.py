"""
轻量级的deepseek token计算方法
pip3 install deepseek-tokenizer
"""

from deepseek_tokenizer import ds_token

# Sample text
text = '''哪吒2的票房是多少'''

# Encode text
result = ds_token.encode(text)

# Print result
print(len(result))
