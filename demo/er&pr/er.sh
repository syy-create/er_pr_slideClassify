MY_Data_Name=er
My_Model_Use=AB-MIL
MY_Classes=2
MY_Device=0
MY_Type=original
MY_SlidePath=/media/data/slide/er

#echo '----------extract feature----------'
#python /home/ipmi2023-sc/PycharmProjects/SlideClassify_stardard20240310/extract_feature_clean.py \
#--datasetsName $MY_Data_Name \
#--model $My_Model_Use \
#--slide_dir $MY_SlidePath \
#--batch_size 256 \
#--workers 8 \
#--device $MY_Device \
#--round 0 \
#--n_classes $MY_Classes
#
echo "----------training MIL model----------"
python /home/Projects/SlideClassify/train_model.py \
--datasetsName $MY_Data_Name \
--n_classes $MY_Classes \
--model $My_Model_Use \
--k 0 \
--device $MY_Device \
--round 0 \
--lr 2e-4

iteration=(1 2 )
for num in "${iteration[@]}"
do
    echo "----------pseudoLabeling----------"
    python /home/Projects/SlideClassify/PseudoLabeling.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --n_classes $MY_Classes \
    --round $num

    echo "----------creating patches----------"
    python /home/Projects/SlideClassify/M1-1_create_patch.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --slide_dir $MY_SlidePath \
    --n_classes $MY_Classes \
    --type $MY_Type \
    --round $num

    echo "----------update feature extraction----------"
    python /home/Projects/SlideClassify/M1-2_updata_model.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --device $MY_Device \
    --n_classes $MY_Classes \
    --round $num

    echo "----------extracting feature----------"
    python /home/Projects/SlideClassify/extract_feature_clean.py \
    --datasetsName $MY_Data_Name \
    --model $My_Model_Use \
    --slide_dir $MY_SlidePath \
    --device $MY_Device \
    --n_classes $MY_Classes \
    --round $num

    echo "----------training MIL model----------"
    python /home/Projects/SlideClassify/train_model.py \
    --datasetsName $MY_Data_Name \
    --n_classes $MY_Classes \
    --model $My_Model_Use \
    --k 0 \
    --device $MY_Device \
    --round $num
done
