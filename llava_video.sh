export NUMEXPR_MAX_THREADS=$(cat /sys/fs/cgroup/cpu.max | awk '{print $1/$2}')


# ours
for cost in 0.3 ; do
for event_upper_bound in 0.2; do
    for similarity_lower_bound in 0.8; do
        for TASK in videomme; do
            python -m accelerate.commands.launch \
                --num_processes=2 \
                -m lmms_eval \
                --model llava_video \
                --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=$similarity_lower_bound,event_upper_bound=$event_upper_bound,merge_type=new_topk,right=True,bottom=True,spatial=True,temporal=True,strategy=3,mm_spatial_pool_mode=bilinear\
                --tasks $TASK \
                --batch_size 1 \
                --output_path ./logs/ 
        done 
    done
done
done 