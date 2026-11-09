---
{{card_data}}
---

# {{ model_name | default("待补充模型名称", true) }} 模型说明

本策略使用 [LeRobot](https://github.com/huggingface/lerobot) 工具链。

- 许可证：{{ license | default("待补充", true) }}
{% if base_model %}- 底座模型：[{{ base_model }}](https://huggingface.co/{{ base_model }})
{% endif %}{% if robot_type %}- 机器人类型：{{ robot_type }}
{% endif %}

## 输入和输出

{% if input_features %}{% for name, feature in input_features.items() %}- 输入 `{{ name }}`：{{ feature.type.value }}，形状 {{ feature.shape }}。
{% endfor %}{% endif %}{% if output_features %}{% for name, feature in output_features.items() %}- 输出 `{{ name }}`：{{ feature.type.value }}，形状 {{ feature.shape }}。
{% endfor %}{% endif %}

## 训练与验证

请补充数据来源、训练参数、验证结果、使用限制和原始方法引用。
工具链说明见 [LeRobot 文档](https://huggingface.co/docs/lerobot/index)。
