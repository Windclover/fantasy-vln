from typing import List, Dict, Any, Optional

from swift.llm.template.template.qwen import Qwen2_5VLTemplate

"""
只修改了 encode() 和 _data_collator()
因为原版 template 处理“一条普通 sample”，而 FantasyVLN 一条 sample 里面有四个 branch。
"""

class MyQwen2_5VLTemplate(Qwen2_5VLTemplate):
    """
    进入前：
    inputs = {
        "Non_CoT": {
            "messages": non_messages,
            "images": images
        },

        "T_CoT": {
            "messages": tcot_messages,
            "images": images
        },

        "V_CoT": {
            "messages": vcot_messages,
            "images": images
        },

        "MM_CoT": {
            "messages": mmcot_messages,
            "images": images
        }
    }
    进入后：
    encoded_inputs = {
        "Non_CoT": {
            "input_ids": ...,
            "labels": ...,
            "pixel_values": ...,
            ...
        },

        "T_CoT": {
            "input_ids": ...,
            "labels": ...,
            "pixel_values": ...,
            ...
        },

        "V_CoT": {
            "input_ids": ...,
            "labels": ...,
            "pixel_values": ...,
            ...
        },

        "MM_CoT": {
            "input_ids": ...,
            "labels": ...,
            "pixel_values": ...,
            ...
        }
    }
    """
    my_qwen2_5_vl = 'my_qwen2_5_vl'

    def encode(
        self,
        inputs: Dict[str, Any],
        return_template_inputs: bool = False,
        return_length: bool = False
    ) -> Dict[str, Any]:
        encoded_inputs = {}
        for branch_name, branch_input in inputs.items():
        # encoded_inputs["Non_CoT"] =
        #   Qwen2_5VLTemplate.encode(
        #     inputs["Non_CoT"]
        #   )
            encoded_inputs[branch_name] = super().encode(
                branch_input,
                return_template_inputs=return_template_inputs,
                return_length=return_length
            )
        return encoded_inputs
    
    def _data_collator(
        self,
        batch: List[Dict[str, Any]],
        *,
        padding_to: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        输入：
        batch
            ├── sample 0
            │   ├── Non
            │   ├── T
            │   ├── V
            │   └── MM
            │
            └── sample 1
                ├── Non
                ├── T
                ├── V
                └── MM
        最终结构：
        batch
            ├── Non_CoT
            │   ├── sample0
            │   └── sample1
            │
            ├── T_CoT
            │   ├── sample0
            │   └── sample1
            │
            ...
        
        """
        collated_batch = {}
        for branch_name in batch[0].keys():
            branch_batch = []
            for sample in batch:
                branch_batch.append(sample[branch_name])

            # 交给父类原版 Qwen2.5-VL collator，
            # 父类 collator 会负责正常的：padding input_ids、padding labels、整理 attention_mask、整理 image inputs...
            collated_batch[branch_name] = super()._data_collator(branch_batch, padding_to=padding_to)
        return collated_batch
