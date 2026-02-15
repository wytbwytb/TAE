num_heads_list=(8 16 24 32 40 48)
alpha_list=(5 10 15 20 25 30)
device=0

# 外层循环
for num_head in "${num_heads_list[@]}"; do
  # 内层循环
  for alpha in "${alpha_list[@]}"; do
    # 执行命令
    echo $num_head, $alpha
    python validation/validate_2fold.py --model_name llama3_instruct_8B --num_heads $num_head \
    --alpha $alpha --device $device --num_fold 2 --use_center_of_mass --save_subfix _average_gcnmi_0
  done

done


