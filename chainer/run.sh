./train_imagenet.py --arch alexnet        --batchsize 128     | tee out_alexnet.log
./train_imagenet.py --arch googlenet      --batchsize 128     | tee out_googlenet.log
./train_imagenet.py --arch vgga           --batchsize 64      | tee out_vgga.log
./train_imagenet.py --arch overfeat       --batchsize 128     | tee out_overfeat.log
./train_imagenet.py --arch alexnet_fp16   --batchsize 128     | tee out_alexnet_fp16.log
./train_imagenet.py --arch googlenet_fp16 --batchsize 128     | tee out_googlenet_fp16.log
./train_imagenet.py --arch vgga_fp16      --batchsize 64      | tee out_vgga_fp16.log
./train_imagenet.py --arch overfeat_fp16  --batchsize 128     | tee out_overfeat_fp16.log
